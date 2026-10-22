# 레이어 민감도 2단계: fc2(mlp/c_proj) 를 블록 하나씩 FP32 로 남김 (12개, 약 70분)
# 1단계에서 S-fc2 100% 회복, S-d4-7 102% 회복 -> 블록 4~7 의 fc2 중 어디인지 좁힌다
export KIND=fc2
bash setup_and_run.sh sens2
