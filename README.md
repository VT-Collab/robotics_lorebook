# robotics lorebook

## Dependencies

```
pip install numpy sentence_transformers openai termcolor PyYaml pybullet imageio[ffmpeg]
```

Or simply run

```
uv init
uv sync
```

Then for any python files, run

```
uv run python *.py
```

## Execution

```
export ARC_API_KEY=YOUR_API_KEY_HERE
python main.py
```

To provide feedback for a subtask, ctrl+c during execution (or the 5 second grace period after execution)
