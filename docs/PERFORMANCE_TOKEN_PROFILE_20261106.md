# Performance and Token profile (2026-11-06)

## Evidence source

This profile uses the independently frozen four-case composite-request
regression at commit `b01c7ce6140a817a00ab933f8dbb7a32841b3812`.
The run used `qwen3.6-plus`, zero retries, zero model switches and zero Judge
calls. All 19 provider calls returned usage data.

Retrospective validation found that case 048 reused a conversation namespace
from an earlier failed attempt, so this component profile is observational and
must not be presented as a clean isolated benchmark. A formal A/B run requires
fresh, non-overlapping conversation namespaces for both groups.

## Observed breakdown

| Component | Calls | Tokens | Share | Average call latency |
|---|---:|---:|---:|---:|
| Intent recognizer | 4 | 11,452 | 29.5% | 36.37 s |
| Evidence selector | 7 | 6,657 | 17.2% | 14.05 s |
| Profile extraction | 4 | 4,554 | 11.7% | 18.10 s |
| Final Agent generation | 4 | 16,116 | 41.6% | 29.11 s |
| **Total** | **19** | **38,779** | **100%** | — |

The structured subtasks (intent, evidence selection and profile extraction)
accounted for 22,663 tokens, or 58.4% of observed usage. The intent calls asked
for at most 256 response tokens but reported 7,914 output tokens in total. This
is consistent with Qwen 3.6's provider-default thinking mode, where reasoning
tokens are billed as output tokens and are not necessarily constrained by the
legacy `max_tokens` behavior.

## First optimization

For Qwen models, `tracked_messages_create` now sends
`thinking: {"type": "disabled"}` only for:

- `intent_recognizer`
- `evidence_selector`
- `memory.profile`

Final Agent generation retains the provider default so that this first change
does not simultaneously alter response-generation quality. Callers may override
the policy through `extra_body`, and deployments may set
`ECHOMIND_DISABLE_THINKING_COMPONENTS` to a different comma-separated list.

## Claim boundary

The 58.4% figure is the share of tokens in components affected by the change,
not a measured saving from the exploratory profile.

## Verified isolated A/B (confirmed 2026-11-07)

A subsequent paired run used commit
`d097b2bd6b55679f5058ea66a0ffd73ca815a626`, fresh non-overlapping Redis
conversation namespaces, the same four questions, corpus and
`qwen3.6-plus` alias, zero retries, zero model switches and zero Judge calls.
The only treatment change was disabling thinking for the three structured
components listed above. Final Agent generation retained the provider default.

| Metric | Control | Treatment | Change |
|---|---:|---:|---:|
| Provider calls | 19 | 19 | 0 |
| Input tokens | 16,913 | 17,049 | +0.8% |
| Output tokens | 19,904 | 5,050 | -74.6% |
| Total tokens | 36,817 | 22,099 | **-40.0%** |
| Mean end-to-end latency | 82.22 s | 34.77 s | **-57.7%** |
| Median end-to-end latency | 77.37 s | 36.96 s | **-52.2%** |

The adjusted structured components fell from 22,460 to 7,617 tokens
(-66.1%), while Final Agent usage remained effectively unchanged (14,357 vs
14,482 tokens). The project owner reviewed both variants and confirmed that all
four paired target tasks passed.

These are four fixed composite-request cases, not full-set accuracy, a
production SLA, or a guaranteed saving on unseen traffic. The fixed execution
order (control before treatment) also means provider load may contribute to the
latency difference; the Token attribution is more direct than the latency
estimate.
