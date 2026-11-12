#!/bin/bash
tmux kill-session -t glove 2>/dev/null
tmux new-session -d -s glove -n encoder "bash -lc 'source /opt/ros/jazzy/setup.bash && source ~/haptic_ws/install/setup.bash && ros2 run haptic_encoder haptic_encoder; exec bash'"
tmux new-window -t glove -n vision "bash -lc 'source /opt/ros/jazzy/setup.bash; while true; do python3 ~/integrated_safety.py; echo restart; sleep 2; done'"
sleep 20
ss -ltnp | grep -q 5005 && echo "READY: 5005 open" || echo "NOT READY: tmux attach -t glove"
