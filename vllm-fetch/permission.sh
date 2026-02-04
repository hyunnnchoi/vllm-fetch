# setup_permissions.sh (sudo로 실행, 일회성)
#!/bin/bash
sudo chown -R $USER:$USER /vllm
sudo chown -R $USER:$USER /lmcache
echo "Ownership changed for $USER"
