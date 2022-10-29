#!/bin/bash
sleep 3
PID=$(ps -e|grep mv_driver_node|awk '{printf $1}')
kill -9 ${PID}
exit 0