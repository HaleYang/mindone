# 修改
- 所有的自定义Pipelines均放在该目录下并规范命名（无需限制为`pipeline.py`）
- 参考`mindone/diffusers/utils/dynamic_modules_utils.py`中的修改，注意修改覆盖到所有逻辑分支
- 实际调用时候，`custom_pipeline="path/to/specific_custom_pipeline.py"`

# 调用
```py
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import mindspore
        >>> from mindone.diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers",
        ...     custom_pipeline="./sd3_pipeline.py",
        ...     mindspore_dtype=mindspore.float16,
        ... )
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt, use_cache_and_tgate=True)[0][0]
        >>> image.save("sd3.png")
        ```
"""
```