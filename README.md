# panda-LM-test
internal code for testing LM robot control

## updates

- fixed move_to_pose so the robot actually goes where it is supposed to
	[solution was to remove internal collisions and joint limits, fine for simulation]
- updated the prompt and settings in qwen to remove internal memory
	[settings changed to reflect this, can delete chats occationally to make sure]
- used Qwen, Gemini to refine the prompt
	[how would you format this, how would you improve this, any parts unclear, etc.]
- added parameterized functions (skills?) for different types of grasps
	[side grasp, top down grasp, etc.]
- my impression is that it is very close, and just needs minor tuning to complete tasks.
