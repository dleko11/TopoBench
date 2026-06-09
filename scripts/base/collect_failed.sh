#!/bin/bash

show_usage() {
    echo "Usage: $0 --path /path/to/FAILED_RUNS.log --output_file /path/to/output_script.sh"
    echo ""
    echo "Options:"
    echo "  --path           Required path to FAILED_RUNS.log."
    echo "  --output_file    Required path to the rerun script to create."
    echo "  -h, --help       Show this help message."
}

INPUT_LOG_FILE=""
OUTPUT_SCRIPT_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --path)
            if [ -z "$2" ]; then
                echo "Error: missing value for --path" >&2
                show_usage
                exit 1
            fi
            INPUT_LOG_FILE="$2"
            shift 2
            ;;
        --output_file)
            if [ -z "$2" ]; then
                echo "Error: missing value for --output_file" >&2
                show_usage
                exit 1
            fi
            OUTPUT_SCRIPT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_LOG_FILE" ]; then
    echo "Error: --path argument is required." >&2
    show_usage
    exit 1
fi

if [ -z "$OUTPUT_SCRIPT_FILE" ]; then
    echo "Error: --output_file argument is required." >&2
    show_usage
    exit 1
fi

if [ ! -f "$INPUT_LOG_FILE" ]; then
    echo "Error: input file not found: $INPUT_LOG_FILE" >&2
    exit 1
fi

echo "Collecting failed runs from $INPUT_LOG_FILE..."

echo "#!/bin/bash" > "$OUTPUT_SCRIPT_FILE"
echo "set -e" >> "$OUTPUT_SCRIPT_FILE"
echo "" >> "$OUTPUT_SCRIPT_FILE"

command_count=0
while read -r line; do
    if [[ "$line" == "Command: "* ]]; then
        command_to_run="${line#Command: }"
        escaped_command=$(printf "%s" "$command_to_run" | sed "s/'/'\\\\''/g")
        echo "echo '================================='" >> "$OUTPUT_SCRIPT_FILE"
        echo "echo 'Rerunning: $escaped_command'" >> "$OUTPUT_SCRIPT_FILE"
        echo "echo '---------------------------------'" >> "$OUTPUT_SCRIPT_FILE"
        echo "$command_to_run" >> "$OUTPUT_SCRIPT_FILE"
        echo "echo 'SUCCESS - Command finished.'" >> "$OUTPUT_SCRIPT_FILE"
        echo "echo ''" >> "$OUTPUT_SCRIPT_FILE"
        ((command_count++))
    fi
done < "$INPUT_LOG_FILE"

chmod +x "$OUTPUT_SCRIPT_FILE"

echo "Done."
echo "Collected $command_count commands into $OUTPUT_SCRIPT_FILE"
