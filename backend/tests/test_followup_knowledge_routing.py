"""口语连接与换行请求的开发回归，使用工具和选择器替身。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import BaseAgent, BillingAgent, Request
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory
from core.knowledge_routing import scope_knowledge_query, split_knowledge_queries


@pytest.mark.parametrize('message,expected', [
    ('密码忘了怎么重置还有电子发票多久能开好', ['密码忘了怎么重置', '电子发票多久能开好']),
    ('登录提示401顺便问100积分能抵多少钱', ['登录提示401', '100积分能抵多少钱']),
    ('加急配送多久到\n退款审核通过后多久能到账', ['加急配送多久到', '退款审核通过后多久能到账']),
    ('如何重置密码还想问一下电子发票何时开好', ['如何重置密码', '电子发票何时开好']),
    ('登录401顺便再问一下订单异常应提供什么', ['登录401', '订单异常应提供什么']),
    ('登录401\r\n退款审核多久\r\n电子发票何时开好', ['登录401', '退款审核多久', '电子发票何时开好']),
    ('密码忘了怎么重置？还有，发票申请后多久能开好？', ['密码忘了怎么重置', '发票申请后多久能开好']),
    ('密码忘了怎么重置？顺便问一下，电子发票何时开好？', ['密码忘了怎么重置', '电子发票何时开好']),
    ('登录提示401。还想问一下：订单异常应提供什么', ['登录提示401', '还想问一下：订单异常应提供什么']),
])
def test_explicit_followup_goals_are_split_with_their_fields(message, expected):
    assert split_knowledge_queries(message) == expected


@pytest.mark.parametrize('message', [
    '退款审核还有几天能完成',
    '发票上还有退款信息',
    '标准配送多久到还有加急配送多久到',
    '密码忘了怎么重置还有今天天气怎么样',
    '登录401订单重复扣款99元',
    '退款审核多久，审核通过后多久能到账？',
    '登录401还有退款多久还有积分抵多少还有发票多久',
    '退款到账前还要验证密码吗',
    '密码忘了怎么重置？还有，今天天气怎么样？',
    '退款审核还有几天能完成？还有，具体是哪一天？',
    '密码忘了怎么重置？还有，发票上还有退款信息吗？以及我该联系谁？',
])
def test_references_single_topics_and_ambiguous_goals_keep_original(message):
    assert split_knowledge_queries(message) == [message]


@pytest.mark.parametrize('agent,expected', [
    ('technical', '登录提示401'),
    ('billing', '订单被重复扣款99元'),
])
def test_domain_scope_retains_error_code_and_amount(agent, expected):
    assert scope_knowledge_query('登录提示401还有订单被重复扣款99元', agent) == expected


def test_domain_scope_does_not_drop_an_unclassified_third_goal():
    message = '登录提示401\n退款审核多久\n请说明你支持哪些功能'
    assert scope_knowledge_query(message, 'technical') == message
    assert scope_knowledge_query(message, 'billing') == message


class LocalRag:
    def __init__(self):
        self.queries = []

    async def call(self, name, params, use_cache=True):
        query = params['query']
        self.queries.append(query)
        if '密码' in query:
            document = {'title': '账户安全', 'content': '可通过绑定手机号或邮箱重置密码。'}
        elif '退款' in query:
            document = {'title': '退款政策', 'content': '退款申请提交后1-3个工作日审核。'}
        else:
            document = {'title': '技术故障排查', 'content': '401表示认证失败，应检查凭证。'}
        return SimpleNamespace(success=True, data=[document], reranked=False)

    async def search_with_rewrite(self, *args, **kwargs):
        raise AssertionError('本地命中或明确知识缺口不应调用查询改写')


class Selector:
    def __init__(self, fail_password=False):
        self.calls = []
        self.fail_password = fail_password

    async def select(self, query, candidates):
        self.calls.append((query, [item['title'] for item in candidates]))
        if self.fail_password and '密码' in query:
            raise TimeoutError('test selector timeout')
        return {'success': True, 'supported': True, 'selected_title': candidates[0]['title'],
                'evidence': '测试文档直接支持当前子问题'}


def search(message, intent, selector):
    manager = LocalRag()
    request = Request(message=message, user_id='followup-test', conv_id='test',
                      intent=intent, intent_confidence=0.95)
    tool = build_shared_rag_tools(manager, selector)['search_knowledge_base']
    result = asyncio.run(tool.handler(request, {'query': message, 'top_k': 3}))
    return result, manager


@pytest.mark.parametrize('connector', ['还有', '顺便问', '\n', '？还有，', '？顺便问一下，'])
def test_supported_password_and_invoice_gap_reach_generation_separately(connector):
    first, second = '密码忘了怎么重置', '电子发票何时开好'
    message = first + connector + second
    selector = Selector()
    result, manager = search(message, IntentCategory.INVOICE, selector)
    assert result['evidence_gate'] == 'composite_partial'
    assert result['evidence_complete'] is False
    assert [part['supported'] for part in result['subquery_results']] == [True, False]
    assert [part['query'] for part in result['subquery_results']] == [first, second]
    assert [row['title'] for row in result['results']] == ['账户安全']
    assert manager.queries == [first, second]
    assert selector.calls == [(first, ['账户安全'])]
    payload = BaseAgent._build_grounded_evidence_payload(message, result)
    assert [part['evidence_status'] for part in payload['subquery_evidence']] == ['accepted', 'insufficient']


def test_two_supported_goals_keep_both_sources():
    selector = Selector()
    result, _ = search('登录401还有退款审核多久', IntentCategory.TECHNICAL_LOGIN, selector)
    assert result['evidence_gate'] == 'composite_complete'
    assert [row['title'] for row in result['results']] == ['技术故障排查', '退款政策']
    assert len(selector.calls) == 2


def test_selector_timeout_only_blocks_its_own_goal():
    selector = Selector(fail_password=True)
    message = '密码忘了怎么重置顺便问退款审核多久'
    result, _ = search(message, IntentCategory.REFUND, selector)
    assert result['evidence_gate'] == 'composite_partial'
    assert [part['supported'] for part in result['subquery_results']] == [False, True]
    assert result['subquery_results'][0]['evidence_gate'] == 'semantic_selector_failed'
    assert [row['title'] for row in result['results']] == ['退款政策']


@pytest.mark.parametrize('message', [
    '密码忘了怎么重置？还有，发票申请后多久能开好？',
    '密码忘了怎么重置？顺便问一下，发票申请后多久能开好？',
])
def test_billing_prefetch_keeps_supported_password_when_invoice_has_no_evidence(message):
    """走真实 Agent 预取和证据注入路径；远程生成仅使用可检查的替身。"""
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type='text', text='fixture generation')],
            stop_reason='end_turn',
        )

    manager = LocalRag()
    selector = Selector()
    agent = BillingAgent(SimpleNamespace(messages=SimpleNamespace(create=create)), 'test-model')
    agent.set_shared_tools(build_shared_rag_tools(manager, selector))
    req = Request(message=message, user_id='punctuation-test', conv_id='isolated',
                  intent=IntentCategory.INVOICE, intent_confidence=0.93)
    response = asyncio.run(agent.handle(req))
    assert response.success
    assert response.content == 'fixture generation'
    assert len(calls) == 1, '有据子问题应进入生成阶段，不能被发票知识缺口整体关闭'
    evidence_message = next(
        item['content'] for item in calls[0]['messages']
        if isinstance(item.get('content'), str) and item['content'].startswith('[已验证知识证据]\n')
    )
    evidence = json.loads(evidence_message.split('\n', 1)[1])
    assert [doc['title'] for doc in evidence['selected_documents']] == ['账户安全']
    assert [part['evidence_status'] for part in evidence['subquery_evidence']] == ['accepted', 'insufficient']
    assert [part['question'] for part in evidence['subquery_evidence']] == [
        '密码忘了怎么重置', '发票申请后多久能开好',
    ]
    assert response.tool_traces[0]['evidence_gate'] == 'composite_partial'
