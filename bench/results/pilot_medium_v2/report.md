# qddot benchmark report

| Agent | Condition | Success | Fuzz pass rate | False-done rate | Tokens in/out | Cost USD | Wall seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| claude_code:claude-haiku-5-5 | none | 60.0% | 64.3% | 20.0% | 320/426027 | 0.4556 | 2638.31 |
| claude_code:claude-haiku-5-5 | qddot | 93.3% | 93.0% | 6.7% | 444/188204 | 0.2565 | 1326.43 |
| claude_code:claude-haiku-5-5 | stdout | 100.0% | 100.0% | 0.0% | 530/318610 | 0.4720 | 2048.43 |
| codex:gpt-6-luna | none | 40.0% | 39.7% | 6.7% | 2201850/79499 | 0.0000 | 1012.73 |
| codex:gpt-6-luna | qddot | 86.7% | 86.3% | 13.3% | 12746034/136867 | 0.0000 | 1974.86 |
| codex:gpt-6-luna | stdout | 73.3% | 73.3% | 0.0% | 4488147/77470 | 0.0000 | 1188.61 |
