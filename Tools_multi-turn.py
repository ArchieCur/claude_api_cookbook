from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from datetime import datetime, timedelta
from anthropic.types import ToolParam, Message

# Load env variables and create client
client = Anthropic()
model = "claude-3-5-haiku-20241022"  # Claude 3.5 Haiku (fast and cost-effective)


# ============================================
# Helper Functions
# ============================================

def add_user_message(messages, message):
    """Add a user message to the conversation history.

    Handles both raw strings and Message objects for flexibility.
    """
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(user_message)


def chat(messages, system=None, temperature=1.0, stop_sequences=[], tools=None):
    """Unified function for making API calls to Claude.

    Handles both regular conversations and tool-enabled conversations.
    """
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "stop_sequences": stop_sequences,
    }

    if tools:
        params["tools"] = tools

    if system:
        params["system"] = system

    message = client.messages.create(**params)
    return message


def text_from_message(message):
    """Extract all text content from a message response.

    Filters out non-text blocks (like tool_use blocks) and joins text blocks.
    """
    return "\n".join(
        [block.text for block in message.content if block.type == "text"]
    )


# ============================================
# Tool Functions
# ============================================

def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)


def add_duration_to_datetime(
    datetime_str, duration=0, unit="days", input_format="%Y-%m-%d"
):
    date = datetime.strptime(datetime_str, input_format)

    if unit == "seconds":
        new_date = date + timedelta(seconds=duration)
    elif unit == "minutes":
        new_date = date + timedelta(minutes=duration)
    elif unit == "hours":
        new_date = date + timedelta(hours=duration)
    elif unit == "days":
        new_date = date + timedelta(days=duration)
    elif unit == "weeks":
        new_date = date + timedelta(weeks=duration)
    elif unit == "months":
        month = date.month + duration
        year = date.year + month // 12
        month = month % 12
        if month == 0:
            month = 12
            year -= 1
        day = min(
            date.day,
            [
                31,
                29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                31,
                30,
                31,
                30,
                31,
                31,
                30,
                31,
                30,
                31,
            ][month - 1],
        )
        new_date = date.replace(year=year, month=month, day=day)
    elif unit == "years":
        new_date = date.replace(year=date.year + duration)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

    return new_date.strftime("%A, %B %d, %Y %I:%M:%S %p")


def set_reminder(content, timestamp):
    print(f"----\nSetting the following reminder for {timestamp}:\n{content}\n----")
    return f"Reminder set for {timestamp}"


# ============================================
# Tool Schemas
# ============================================

get_current_datetime_schema = ToolParam({
    "name": "get_current_datetime",
    "description": "Get the current date and time formatted according to a specified format string. Uses Python's strftime format codes (e.g., '%Y-%m-%d %H:%M:%S' for '2024-01-15 14:30:45', '%B %d, %Y' for 'January 15, 2024', '%I:%M %p' for '02:30 PM'). Common format codes: %Y (4-digit year), %m (month 01-12), %d (day 01-31), %H (hour 00-23), %M (minute 00-59), %S (second 00-59), %B (full month name), %A (full weekday name), %I (hour 01-12), %p (AM/PM).",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {
                "type": "string",
                "description": "The strftime format string to use for formatting the datetime. Must be a non-empty string following Python strftime conventions. Examples: '%Y-%m-%d %H:%M:%S' (default), '%B %d, %Y at %I:%M %p', '%A, %B %d, %Y', '%Y/%m/%d'.",
                "default": "%Y-%m-%d %H:%M:%S"
            }
        },
        "required": []
    }
})

add_duration_to_datetime_schema = ToolParam({
    "name": "add_duration_to_datetime",
    "description": "Adds a specified duration to a datetime string and returns the resulting datetime in a detailed format. This tool converts an input datetime string to a Python datetime object, adds the specified duration in the requested unit, and returns a formatted string of the resulting datetime. It handles various time units including seconds, minutes, hours, days, weeks, months, and years, with special handling for month and year calculations to account for varying month lengths and leap years. The output is always returned in a detailed format that includes the day of the week, month name, day, year, and time with AM/PM indicator (e.g., 'Thursday, April 03, 2025 10:30:00 AM').",
    "input_schema": {
        "type": "object",
        "properties": {
            "datetime_str": {
                "type": "string",
                "description": "The input datetime string to which the duration will be added. This should be formatted according to the input_format parameter.",
            },
            "duration": {
                "type": "number",
                "description": "The amount of time to add to the datetime. Can be positive (for future dates) or negative (for past dates). Defaults to 0.",
            },
            "unit": {
                "type": "string",
                "description": "The unit of time for the duration. Must be one of: 'seconds', 'minutes', 'hours', 'days', 'weeks', 'months', or 'years'. Defaults to 'days'.",
            },
            "input_format": {
                "type": "string",
                "description": "The format string for parsing the input datetime_str, using Python's strptime format codes. For example, '%Y-%m-%d' for ISO format dates like '2025-04-03'. Defaults to '%Y-%m-%d'.",
            },
        },
        "required": ["datetime_str"],
    },
})

set_reminder_schema = ToolParam({
    "name": "set_reminder",
    "description": "Creates a timed reminder that will notify the user at the specified time with the provided content. This tool schedules a notification to be delivered to the user at the exact timestamp provided. It should be used when a user wants to be reminded about something specific at a future point in time. The reminder system will store the content and timestamp, then trigger a notification through the user's preferred notification channels (mobile alerts, email, etc.) when the specified time arrives. Reminders are persisted even if the application is closed or the device is restarted. Users can rely on this function for important time-sensitive notifications such as meetings, tasks, medication schedules, or any other time-bound activities.",
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message text that will be displayed in the reminder notification. This should contain the specific information the user wants to be reminded about, such as 'Take medication', 'Join video call with team', or 'Pay utility bills'.",
            },
            "timestamp": {
                "type": "string",
                "description": "The exact date and time when the reminder should be triggered, formatted as an ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS) or a Unix timestamp. The system handles all timezone processing internally, ensuring reminders are triggered at the correct time regardless of where the user is located. Users can simply specify the desired time without worrying about timezone configurations.",
            },
        },
        "required": ["content", "timestamp"],
    },
})


# ============================================
# Tool Registry and Processing
# ============================================

# Map tool names to actual Python functions
TOOL_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
    "add_duration_to_datetime": add_duration_to_datetime,
    "set_reminder": set_reminder,
}

# All available tool schemas
ALL_TOOLS = [
    get_current_datetime_schema,
    add_duration_to_datetime_schema,
    set_reminder_schema,
]


def process_tool_call(tool_name, tool_input):
    """Execute the tool function and return the result with logging.

    Returns:
        tuple: (result_string, is_error_boolean)
    """
    # Tool usage logging - helps track which tools are being used
    print(f"[TOOL USED]: {tool_name}")
    print(f"[INPUT]: {tool_input}")

    if tool_name not in TOOL_FUNCTIONS:
        error_msg = f"Error: Unknown tool '{tool_name}'"
        print(f"[ERROR]: {error_msg}")
        return error_msg, True

    try:
        tool_function = TOOL_FUNCTIONS[tool_name]
        result = tool_function(**tool_input)
        print(f"[OUTPUT]: {result}")
        return str(result), False
    except Exception as e:
        error_msg = f"Error executing {tool_name}: {str(e)}"
        print(f"[ERROR]: {error_msg}")
        return error_msg, True


# ============================================
# Multi-turn Tool Conversation Loop
# ============================================

def chat_with_tools(user_message):
    """
    Main conversation loop that handles multi-turn tool use.
    Continues until Claude provides a final text response.
    """
    # Initialize conversation
    messages = [{
        "role": "user",
        "content": user_message
    }]

    print(f"User: {user_message}\n")

    # Loop until Claude stops requesting tools
    while True:
        # Call Claude with all available tools
        response = chat(messages, tools=ALL_TOOLS)

        # Append assistant's response to conversation history
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # Check if Claude wants to use any tools
        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

        if not tool_use_blocks:
            # No tool use - Claude is done, extract final text response
            final_response = text_from_message(response)
            print(f"Assistant: {final_response}\n")
            return final_response

        # Process each tool call
        tool_results = []
        for tool_use in tool_use_blocks:
            tool_name = tool_use.name
            tool_input = tool_use.input

            # Execute the tool (includes logging)
            result, is_error = process_tool_call(tool_name, tool_input)
            print()  # Blank line for readability

            # Collect tool result
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
                "is_error": is_error,
            })

        # Send tool results back to Claude
        messages.append({
            "role": "user",
            "content": tool_results
        })


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    # Example 1: Simple tool use
    print("=" * 60)
    print("Example 1: Get current time")
    print("=" * 60)
    chat_with_tools("What time is it right now?")

    print("\n" + "=" * 60)
    print("Example 2: Multi-step calculation")
    print("=" * 60)
    chat_with_tools("What's the date 7 days from now? Then set a reminder for that date to 'Review weekly goals'")
