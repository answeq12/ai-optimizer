# E37 준비: WSL 디스크·D 드라이브 여유 확인
df -h $HOME /mnt/d
du -sh $(cat logs/da_feat_dir.txt 2>/dev/null || cat ../logs/da_feat_dir.txt)
ls -la $(cat ../logs/da_feat_dir.txt 2>/dev/null || cat logs/da_feat_dir.txt)
free -g
