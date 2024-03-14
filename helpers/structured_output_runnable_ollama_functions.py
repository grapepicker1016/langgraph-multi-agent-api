import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union

from langchain_core.output_parsers import (
    BaseGenerationOutputParser,
    BaseOutputParser,
    JsonOutputParser,
)
from langchain_core.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
    PydanticOutputFunctionsParser,
)
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)

from langchain.output_parsers import (
    JsonOutputKeyToolsParser,
    PydanticOutputParser,
    PydanticToolsParser,
)



def create_structured_output_runnable_custom(
    output_schema: Union[Dict[str, Any], Type[BaseModel]],
    llm: Runnable,
    prompt: Optional[BasePromptTemplate] = None,
    *,
    output_parser: Optional[Union[BaseOutputParser,BaseGenerationOutputParser]] = None,
    # enforce_function_usage: bool = True,
    # return_single: bool = True,
    # **kwargs: Any,
) -> Runnable:
    """Create a runnable for extracting structured outputs using Ollama functions."""

    class _OutputFormatter(BaseModel):
            """Output formatter. Should always be used to format your response to the user."""  # noqa: E501

            output: output_schema  # type: ignore

    function = _OutputFormatter
    
    output_parser = output_parser or PydanticAttrOutputFunctionsParser(
        pydantic_schema=_OutputFormatter, attr_name="output"
    )
    
    model = llm.bind(
        functions=[
            convert_to_openai_function(function)
        ],
        function_call={"name": "_OutputFormatter"}
    )
    
    return prompt | model | output_parser