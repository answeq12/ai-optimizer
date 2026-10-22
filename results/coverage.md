# 연산자 커버리지 (P3.5)

정수 연산 비율 = 실행 직전 그래프에서 행렬곱·합성곱 연산(MatMul·Gemm·Conv 계열, 어텐션의 활성값끼리 곱 포함) 중 정수형인 것의 비율.
정수 커널 시간 비중 = 프로파일에서 정수 커널이 차지한 실행 시간의 비율.

| 항목 | A0 | A1 | A2 | A2m | S-c67-fc2 | S-c567-fc2 | S-c67-pc | S-c67-pc-qop | SPL-b567-k32 | A1-c67-pc |
|---|---|---|---|---|---|---|---|---|---|---|
| 정수 연산 비율 (개수) | 0.0% | 66.2% | 100.0% | 66.2% | 63.5% | 62.2% | 63.5% | 63.5% | 63.6% | 63.5% |
| 정수 커널 시간 비중 | 0.0% | 67.5% | 86.0% | 58.2% | 50.8% | 48.0% | 51.7% | 51.6% | 56.3% | 59.5% |
| 행렬곱·합성곱: 정수 / 실수 | 0 / 74 | 49 / 25 | 74 / 0 | 49 / 25 | 47 / 27 | 46 / 28 | 47 / 27 | 47 / 27 | 49 / 28 | 47 / 27 |
| patch embedding Conv 양자화 | 아니오 | 아니오 | 예 | 아니오 | 아니오 | 아니오 | 아니오 | 아니오 | 아니오 | 아니오 |
| 실행 시 남은 Q/DQ 노드 | 0 | 0 | 106 | 98 | 94 | 92 | 94 | 94 | 98 | 0 |

## 실행 직전 그래프의 연산 종류별 개수

| 연산 | A0 | A1 | A2 | A2m | S-c67-fc2 | S-c567-fc2 | S-c67-pc | S-c67-pc-qop | SPL-b567-k32 | A1-c67-pc |
|---|---|---|---|---|---|---|---|---|---|---|
| Add | 24 | 24 | 0 | 60 | 58 | 57 | 58 | 58 | 63 | 24 |
| Concat | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Conv | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| DequantizeLinear | 0 | 0 | 79 | 49 | 47 | 46 | 47 | 47 | 49 | 0 |
| DynamicQuantizeMatMul | 0 | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 47 |
| FusedMatMul | 12 | 12 | 0 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| Gather | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 7 | 1 |
| Gemm | 48 | 0 | 0 | 0 | 2 | 3 | 2 | 2 | 0 | 2 |
| LayerNormalization | 25 | 25 | 26 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| MatMul | 13 | 12 | 0 | 12 | 12 | 12 | 12 | 12 | 15 | 12 |
| QGemm | 0 | 0 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 0 |
| QLinearAdd | 0 | 0 | 61 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QLinearConcat | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QLinearConv | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QLinearMatMul | 0 | 0 | 61 | 37 | 35 | 34 | 35 | 35 | 37 | 0 |
| QLinearMul | 0 | 0 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QLinearSigmoid | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QLinearSoftmax | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QuantizeLinear | 0 | 0 | 27 | 49 | 47 | 46 | 47 | 47 | 49 | 0 |
| QuickGelu | 12 | 12 | 0 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| ReorderOutput | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Reshape | 169 | 109 | 109 | 109 | 113 | 115 | 113 | 113 | 109 | 113 |
| SkipLayerNormalization | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Softmax | 12 | 12 | 0 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| Split | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 |
| Squeeze | 48 | 48 | 48 | 48 | 48 | 48 | 48 | 48 | 48 | 48 |
| Transpose | 85 | 85 | 99 | 85 | 85 | 85 | 85 | 85 | 85 | 85 |

## 실행 시간 상위 연산

**A0**: Gemm 86.6%, Transpose 3.3%, FusedMatMul 2.1%, MatMul 2.0%, QuickGelu 1.3%, LayerNormalization 1.1%, Softmax 0.8%, Conv 0.7%

**A1**: DynamicQuantizeMatMul 67.5% (정수), Transpose 7.7%, FusedMatMul 5.4%, MatMul 5.0%, QuickGelu 3.2%, LayerNormalization 2.6%, Softmax 1.9%, Conv 1.9%

**A2**: QLinearMatMul 57.7% (정수), QLinearAdd 7.4% (정수), QLinearSoftmax 7.4% (정수), QGemm 6.3% (정수), Transpose 6.1%, QLinearMul 4.9% (정수), LayerNormalization 2.6%, DequantizeLinear 1.9%

**A2m**: QLinearMatMul 52.3% (정수), DequantizeLinear 6.3%, Transpose 6.1%, QGemm 5.9% (정수), Add 5.0%, FusedMatMul 4.9%, MatMul 4.5%, QuantizeLinear 3.3%

**S-c67-fc2**: QLinearMatMul 45.2% (정수), Gemm 10.4%, Transpose 5.8%, DequantizeLinear 5.7%, QGemm 5.6% (정수), Add 4.7%, FusedMatMul 4.6%, MatMul 4.2%

**S-c567-fc2**: QLinearMatMul 42.6% (정수), Gemm 15.1%, Transpose 5.5%, QGemm 5.4% (정수), DequantizeLinear 5.4%, FusedMatMul 4.4%, Add 4.2%, MatMul 4.1%

**S-c67-pc**: QLinearMatMul 45.9% (정수), Gemm 10.3%, QGemm 5.8% (정수), Transpose 5.7%, DequantizeLinear 5.7%, FusedMatMul 4.5%, Add 4.5%, MatMul 4.1%

**S-c67-pc-qop**: QLinearMatMul 46.0% (정수), Gemm 10.3%, QGemm 5.7% (정수), Transpose 5.6%, DequantizeLinear 5.6%, FusedMatMul 4.6%, Add 4.6%, MatMul 4.2%

**SPL-b567-k32**: QLinearMatMul 50.4% (정수), QGemm 6.0% (정수), Transpose 5.9%, DequantizeLinear 5.8%, Add 4.9%, FusedMatMul 4.8%, MatMul 4.6%, Gather 3.8%

**A1-c67-pc**: DynamicQuantizeMatMul 59.5% (정수), Gemm 11.2%, Transpose 7.0%, FusedMatMul 4.9%, MatMul 4.5%, QuickGelu 2.8%, LayerNormalization 2.4%, Softmax 1.8%

