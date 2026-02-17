# robotics lorebook

## User Study

To start the user study, ensure that the fields in `user_study_data/next/config.yml` are correct. 

Then run `python server.py` and open a browser to `localhost:8000`.

The server and main_utils scripts do the following:

1. Read from `user_study_data/next/config.yml` to find the next user ID and task to perform
2. Read from `config/tasks/tasks.yml` to find the *order* of tasks to perform
3. Executes these tasks, saving to `user_study_data/global_lorebook.pkl` and `user_study_data/user_(id)`
4. Once the time is up, the simulation closes at the next opportunity. This writes to `next/config.yml`

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
