from monitor.performance_monitor import PerformanceMonitor


class _OrchestratorStub:
    def get_stats(self):
        return {}


class _ToolManagerStub:
    def get_stats(self):
        return {}


def _monitor(agent_latency_warning_ms: float = 30_000) -> PerformanceMonitor:
    return PerformanceMonitor(
        _OrchestratorStub(),
        _ToolManagerStub(),
        agent_latency_warning_ms=agent_latency_warning_ms,
    )


def test_repeated_threshold_violation_keeps_one_active_alert():
    monitor = _monitor(agent_latency_warning_ms=3_000)

    monitor._check_threshold("agent_avg_ms", 18_000, "billing_0")
    monitor._check_threshold("agent_avg_ms", 19_000, "billing_0")

    summary = monitor.summary()
    assert len(summary["active_alerts"]) == 1
    assert len(monitor._alerts) == 1
    assert summary["active_alerts"][0]["value"] == 19_000


def test_alert_resolves_and_can_trigger_again():
    monitor = _monitor(agent_latency_warning_ms=3_000)

    monitor._check_threshold("agent_avg_ms", 18_000, "billing_0")
    monitor._check_threshold("agent_avg_ms", 2_000, "billing_0")
    assert monitor.summary()["active_alerts"] == []
    assert monitor._alerts[0].resolved is True

    monitor._check_threshold("agent_avg_ms", 20_000, "billing_0")
    assert len(monitor.summary()["active_alerts"]) == 1
    assert len(monitor._alerts) == 2


def test_default_latency_threshold_allows_normal_external_llm_delay():
    monitor = _monitor()

    monitor._check_threshold("agent_avg_ms", 18_000, "billing_0")
    assert monitor.summary()["active_alerts"] == []
    assert monitor._routing_penalty(1.0, 18_000) == 0.0

    monitor._check_threshold("agent_avg_ms", 31_000, "billing_0")
    assert len(monitor.summary()["active_alerts"]) == 1
    assert monitor._routing_penalty(1.0, 31_000) > 0.0
    assert monitor.summary()["thresholds"]["agent_avg_ms"] == 30_000
