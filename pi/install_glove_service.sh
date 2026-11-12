#!/bin/bash
set -e
U=$(whoami)
H=$(eval echo ~"$U")

if [ ! -f "$H/start_glove.sh" ]; then
  echo "ERROR: $H/start_glove.sh not found"
  exit 1
fi

sudo tee /etc/systemd/system/haptic-glove.service >/dev/null <<EOF
[Unit]
Description=Haptic Glove (encoder + vision)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=$U
Environment=HOME=$H
WorkingDirectory=$H
ExecStartPre=/bin/sleep 8
ExecStart=-/bin/bash $H/start_glove.sh
ExecStop=-/usr/bin/tmux kill-session -t glove

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable haptic-glove.service

if ! systemctl is-active --quiet avahi-daemon; then
  sudo apt-get install -y avahi-daemon
fi
sudo systemctl enable --now avahi-daemon

echo
echo "=== installed ==="
echo "auto-start : haptic-glove.service (enabled, runs on every boot)"
echo "hostname   : $(hostname)  ->  ssh $U@$(hostname).local"
echo "commands   : sudo systemctl restart haptic-glove   |   sudo systemctl stop haptic-glove"
