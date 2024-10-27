from vertexai.generative_models import GenerativeModel, FunctionDeclaration, Tool, Content, Part, GenerationConfig, ToolConfig
import json
import uuid  # To generate unique IDs for the tool calls

# Helper class to allow dot-access on dictionaries
class DotDict(dict):
    """A simple class to allow dot notation for dictionary keys."""
    def __getattr__(self, item):
        return self[item]

    def model_dump_json(self):
        """Simulate the model_dump_json behavior by returning a JSON string."""
        serializable_dict = {k: v for k, v in self.items() if self._is_serializable(v)}
        return json.dumps(serializable_dict)

    def _is_serializable(self, value):
        """Helper method to check if a value can be JSON serialized."""
        try:
            json.dumps(value)
            return True
        except (TypeError, OverflowError):
            return False


class GeminiClient:
    def __init__(self, model_id):
        # Initialize the model with the provided model ID
        self.model = GenerativeModel(model_id)

    def _convert_swarm_tool_to_gemini_tool(self, swarm_tool):
        """Convert a Swarm tool (in dict format) to Gemini's Tool and FunctionDeclaration."""
        function_details = swarm_tool['function']

        properties = function_details['parameters'].get('properties', {})
        required = function_details['parameters'].get('required', [])

        function_declaration = FunctionDeclaration(
            name=function_details['name'],
            description=function_details.get('description', ''),
            parameters={
                "type": "object",
                "properties": {
                    key: {"type": prop.get("type"), "description": prop.get("description", "")}
                    for key, prop in properties.items()
                },
                "required": required
            }
        )

        return Tool(function_declarations=[function_declaration])

    def _serialize_function_call(self, function_call):
        """Serialize function call arguments."""
        if isinstance(function_call.args, dict):
            return function_call.args
        else:
            return {
                key: getattr(value, "string_value", value) for key, value in function_call.args.items()
            }

    def _create_function_structure(self, function_call):
        """Manually create the function structure expected by core.py."""
        serialized_function_call = self._serialize_function_call(function_call)
        return DotDict({
            "name": function_call.name,
            "arguments": json.dumps(serialized_function_call)  # Serialize the arguments
        })

    def _create_tool_call(self, function_call):
        """Create a tool call object in the format expected by core.py."""
        return DotDict({
            "id": str(uuid.uuid4()),  # Generate a unique ID for the tool call
            "function": self._create_function_structure(function_call),  # Manually create the function structure
            "type": "function"
        })

    def _convert_message(self, message):
        """Convert a message to a Content object or extract function/tool calls."""
        role_mapping = {
            "assistant": "model",
            "tool": "model"
        }

        role = role_mapping.get(message['role'], message['role'])  # Map to valid roles for Gemini

        if 'function_call' in message:
            return Content(
                role=role,
                parts=[Part.from_text(f"{message['function_call']['name']}({message['function_call']['arguments']})")]
            )
        elif 'tool_name' in message:
            return Content(
                role=role,
                parts=[Part.from_text(message['content'])]
            )
        else:
            # Ensure the message has a "content" field even if it's missing
            return Content(
                role=role,
                parts=[Part.from_text(message.get('content', ''))]  # Default to an empty string if "content" is missing
            )

    @property
    def chat(self):
        gemini_client = self

        class Completions:
            def __init__(self, model):
                self.model = model

            def create(self, **kwargs):
                # Extract the full conversation history
                messages = kwargs.get("messages", [])
                swarm_tools = kwargs.get("tools", [])

                # Filter out 'system' role messages (as Gemini doesn't accept them)
                filtered_messages = [
                    msg for msg in messages if msg["role"] != "system"
                ]

                # Convert Swarm tools into Gemini tools
                gemini_tools = [gemini_client._convert_swarm_tool_to_gemini_tool(tool) for tool in swarm_tools]

                # Convert the conversation history into a list of Content objects, including function calls and tool responses
                conversation_history = [
                    gemini_client._convert_message(message) for message in filtered_messages
                ]

                # Generate the response using the Gemini model with the full conversation history and tools
                response = self.model.generate_content(
                    conversation_history,  # Pass the filtered conversation history
                    tools=gemini_tools  # Include tools for function calling
                )

                # Check if the response contains any candidates
                if not response.candidates:
                    # Handle the blocked response gracefully
                    print("DEBUG: Response blocked by safety filters. No candidates were returned.")
                    return DotDict({
                        "choices": [
                            DotDict({
                                "message": DotDict({
                                    "content": "The response was blocked by safety filters and no candidates were returned.",
                                    "tool_calls": None,
                                    "role": "assistant",
                                    "sender": "assistant"
                                })
                            })
                        ]
                    })

                # Check if a function call was generated
                function_calls = response.candidates[0].function_calls
                if function_calls:
                    function_call = function_calls[0]

                    # Serialize the function call into a format expected by core.py
                    tool_call = gemini_client._create_tool_call(function_call)

                    # Pass the function call information back to the Swarm framework
                    return DotDict({
                        "choices": [
                            DotDict({
                                "message": DotDict({
                                    "function_call": {
                                        "name": function_call.name,
                                        "arguments": json.dumps(gemini_client._serialize_function_call(function_call))
                                    },
                                    "tool_calls": [tool_call],  # This will be processed by the core framework
                                    "role": "assistant",
                                    "content": '',  # Ensure there's always a content field
                                    "sender": "assistant"
                                })
                            })
                        ]
                    })

                # Default case where no function was called, just normal content
                return DotDict({
                    "choices": [
                        DotDict({
                            "message": DotDict({
                                "content": response.candidates[0].content.text,  # Ensure content is always present
                                "tool_calls": None,
                                "role": "assistant",
                                "sender": "assistant"
                            })
                        })
                    ]
                })

        return type('Chat', (), {'completions': Completions(self.model)})()
