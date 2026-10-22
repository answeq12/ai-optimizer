# E25 보조: n=1 위치별 x CLS 가중치(1, 50) 스윕, CIFAR 검증 2000장만 (export 안 함). 약 40분
source ~/venvs/halio/bin/activate
for s in 2 6 9 10 11; do for w in 1 50; do python -m src.replaceme --n 1 --start $s --cls-weight $w --no-export 2>&1 | grep "cls_w"; done; done
