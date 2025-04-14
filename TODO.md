- [x] Add constant gear ratio to servos
- [x] Change control range of servos

- [x] Understand how to add control inputs into the environment observations in the right way
  - For example mask only (real) sensor readings and append control inputs to the observation
- [x] Find better reward for to penalize abrupt changes in control inputs

- [x] RANDOMIZE ENVIRONMENT - Starting pose, joints etc.

- [x] Make camera follow the robot

- [x] ADD TERMINATION CONDITION - FOR FLIPPING
- [ ] Fix bug with saving video when in rgb_array mode
- [x] Check Velocity and Position reference frames

- [x] !!! Reward based on position offset from an ideal trajectory !!!
- [x] ACCOUT FOR HEADING VECTOR AS VELOCITY IS IN THE LOCAL FRAME

REALLY IMPORTANT as sums are analogous to logical OR operations while multiplications to logical ANDs

- [ ] Map rewards to 0-1 range to make them easly multibiable !!!
- [ ] Parametrize environment randomization