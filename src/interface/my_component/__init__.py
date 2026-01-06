import os
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "text_selector",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend")
    _component_func = components.declare_component("text_selector", path=build_dir)

def text_selector(text_area_key: str, key=None):
    component_value = _component_func(text_area_key=text_area_key, key=key, default="")
    return component_value
