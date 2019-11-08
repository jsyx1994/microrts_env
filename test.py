import gym
import rts_wrapper
import  os
env = gym.make("Microrts-v0")

print(rts_wrapper.base_dir_path)
print(os.path.join(rts_wrapper.base_dir_path , 'microrts-master/out/artifacts/microrts_master_jar/microrts-master.jar'))