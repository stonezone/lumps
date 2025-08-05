#!/bin/bash

# LUMPS Web UI Server Management Script
# Usage: ./webui.sh [start|stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$SCRIPT_DIR/web"
PID_FILE="$SCRIPT_DIR/.webui.pid"
LOG_FILE="$SCRIPT_DIR/webui.log"
PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[LUMPS WebUI]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # Running
        else
            rm -f "$PID_FILE"  # Clean up stale PID file
            return 1  # Not running
        fi
    else
        return 1  # Not running
    fi
}

# Generate fresh forecast data
generate_data() {
    print_status "🌊 Generating fresh LUMPS forecast data..."
    cd "$SCRIPT_DIR"
    
    # Run LUMPS analysis with enhanced currents and JSON output
    if python3 lumps.py --output-json --enhanced-currents --days 7 > /tmp/lumps_generation.log 2>&1; then
        print_success "Fresh forecast data generated successfully"
        return 0
    else
        print_error "Failed to generate forecast data"
        echo "Error details:"
        cat /tmp/lumps_generation.log | tail -5
        return 1
    fi
}

# Start the web server
start_server() {
    if is_running; then
        print_warning "Web UI is already running on http://localhost:$PORT"
        return 0
    fi

    if [ ! -d "$WEB_DIR" ]; then
        print_error "Web directory not found: $WEB_DIR"
        return 1
    fi

    # Generate fresh data before starting server
    if ! generate_data; then
        print_error "Cannot start web UI without fresh data"
        return 1
    fi

    print_status "Starting LUMPS Web UI server..."
    
    # Start server in background
    cd "$WEB_DIR"
    nohup python3 -m http.server $PORT > "$LOG_FILE" 2>&1 &
    local server_pid=$!
    
    # Save PID
    echo $server_pid > "$PID_FILE"
    
    # Wait a moment to check if server started successfully
    sleep 2
    
    if is_running; then
        print_success "Web UI started successfully!"
        print_status "🌊 Access LUMPS at: ${BLUE}http://localhost:$PORT${NC}"
        print_status "📄 Server logs: $LOG_FILE"
        print_status "🔄 Use '$0 refresh' to update forecast data"
    else
        print_error "Failed to start web UI"
        return 1
    fi
}

# Stop the web server
stop_server() {
    if ! is_running; then
        print_warning "Web UI is not running"
        return 0
    fi

    local pid=$(cat "$PID_FILE")
    print_status "Stopping LUMPS Web UI (PID: $pid)..."
    
    kill "$pid" 2>/dev/null
    
    # Wait for process to stop
    local count=0
    while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if ps -p "$pid" > /dev/null 2>&1; then
        print_status "Force killing server..."
        kill -9 "$pid" 2>/dev/null
    fi
    
    rm -f "$PID_FILE"
    print_success "Web UI stopped"
}

# Show server status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        print_success "Web UI is running (PID: $pid)"
        print_status "🌊 URL: http://localhost:$PORT"
        print_status "📄 Logs: $LOG_FILE"
        
        # Show recent log entries
        if [ -f "$LOG_FILE" ]; then
            echo ""
            print_status "Recent log entries:"
            tail -5 "$LOG_FILE" | sed 's/^/  /'
        fi
    else
        print_warning "Web UI is not running"
    fi
}

# Restart server
restart_server() {
    print_status "Restarting LUMPS Web UI with fresh data..."
    stop_server
    sleep 1
    start_server  # This will auto-generate fresh data
}

# Refresh data without restarting server
refresh_data() {
    print_status "🔄 Refreshing LUMPS forecast data..."
    if generate_data; then
        print_success "Forecast data refreshed successfully"
        if is_running; then
            print_status "🌊 Refresh your browser to see updated data"
        fi
    else
        print_error "Failed to refresh forecast data"
        return 1
    fi
}

# Show usage
show_usage() {
    echo "LUMPS Web UI Server Management"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start    Generate fresh data and start the web UI server"
    echo "  stop     Stop the web UI server"
    echo "  restart  Restart the web UI server with fresh data"
    echo "  refresh  Update forecast data without restarting server"
    echo "  status   Show server status"
    echo "  logs     Show recent log entries"
    echo ""
    echo "The web UI will be available at: http://localhost:$PORT"
    echo "Note: 'start' and 'restart' automatically generate fresh forecast data"
}

# Show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        print_status "LUMPS Web UI Logs:"
        echo "=================="
        cat "$LOG_FILE"
    else
        print_warning "No log file found"
    fi
}

# Main script logic
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    refresh)
        refresh_data
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        show_usage
        ;;
esac