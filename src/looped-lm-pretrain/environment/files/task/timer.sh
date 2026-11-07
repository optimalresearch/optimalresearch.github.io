#!/bin/bash
# Time left in this episode. The task's healthcheck records the start time and
# the episode length (in seconds) just before the agent starts.
START=$(cat /tmp/timer_start 2>/dev/null || date +%s)
SECONDS_TOTAL=$(cat /tmp/timer_seconds 2>/dev/null || echo 0)
LEFT=$(( START + SECONDS_TOTAL - $(date +%s) ))
if [ "$LEFT" -le 0 ]; then
  echo "Time is up."
else
  printf "Time left: %d:%02d (hours:minutes)\n" $(( LEFT / 3600 )) $(( (LEFT % 3600) / 60 ))
fi
