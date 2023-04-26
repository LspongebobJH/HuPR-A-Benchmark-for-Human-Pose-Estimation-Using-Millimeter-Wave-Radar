#!/bin/bash

# cd scratch place
cd logs/

# Download zip dataset from Google Drive
filename='model_best.pth'
fileid='1Hmi2mw_KuSBCS4PVKI7dGWrRmtpKHkJI'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt