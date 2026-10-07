#!/bin/bash
# 사용법: ./yunaviewer.sh start [이미지폴더] | stop | restart [이미지폴더] | status | log
cd "$(dirname "$0")"
PIDFILE=.yunaviewer.pid
LOG=yunaviewer.log
PY=.venv/bin/python

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

url() {
    "$PY" - <<'EOF'
import configparser
c = configparser.ConfigParser()
c.read("yunaviewer.cfg", encoding="utf-8")
host = c.get("server", "host", fallback="127.0.0.1")
print(f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{c.getint('server', 'port', fallback=8890)}")
EOF
}

start() {
    if running; then
        echo "이미 실행 중 (PID $(cat "$PIDFILE")) - $(url)"
        return
    fi
    [ -x "$PY" ] || { echo "$PY 가 없습니다: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; return 1; }
    setsid nohup "$PY" server.py "$@" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 1
    if running; then
        echo "시작됨 (PID $(cat "$PIDFILE")) - $(url)"
    else
        echo "시작 실패. 로그:"
        tail -n 20 "$LOG"
        rm -f "$PIDFILE"
        return 1
    fi
}

stop() {
    if running; then
        kill "$(cat "$PIDFILE")"
        echo "종료됨 (PID $(cat "$PIDFILE"))"
    else
        echo "실행 중인 프로세스 없음"
    fi
    rm -f "$PIDFILE"
}

status() {
    if running; then
        echo "실행 중 (PID $(cat "$PIDFILE")) - $(url)"
    else
        echo "중지됨"
    fi
}

command="${1:-status}"
shift
case "$command" in
    start) start "$@" ;;
    stop) stop ;;
    restart) stop; sleep 1; start "$@" ;;
    status) status ;;
    log) tail -n 50 -f "$LOG" ;;
    *) echo "사용법: $0 start [이미지폴더] | stop | restart [이미지폴더] | status | log"; exit 1 ;;
esac
