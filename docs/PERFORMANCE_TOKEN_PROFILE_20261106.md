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
not a measured saving. Actual Token reduction, latency change and answer quality
must be established with a controlled rerun of the same four frozen cases.
