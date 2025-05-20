import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./out.csv")

#"n,numberOfSamples,time_cpu_float,time_cpu_double,block_size,time_gpu_float,time_gpu_double,diff_float,diff_double,spdup_float,spdup_double,cpu,gpu"
n = df.columns[0]
time_cpu_float = df.columns[2]
time_cpu_double = df.columns[3]
block_sizes = df.columns[4]
time_gpu_float = df.columns[5]
time_gpu_double = df.columns[6]
cpu = df.columns[11]
gpu = df.columns[12]

sizes = list(set(list(df[n])))
sizes.sort()

for size in sizes:
	plt.clf()

	df_cpu = df[ (df[n]==size) & (df[cpu]==1)]
	df_gpu = df[ (df[n]==size) & (df[gpu]==1)]

	t_float_cpu  = np.array(df_cpu[time_cpu_float])
	t_float_gpu  = np.array(df_gpu[time_gpu_float])
	t_double_cpu = np.array(df_cpu[time_cpu_double])
	t_double_gpu = np.array(df_gpu[time_gpu_double])

	spdup_float = list(t_float_cpu / t_float_gpu)
	spdup_double = list(t_double_cpu / t_double_gpu)
		
	block = list(df_gpu[block_sizes])

	plt.plot(block, spdup_float, label=f"float")
	plt.plot(block, spdup_double, label=f"double")

	plt.xlabel("block size")
	plt.ylabel("speedup")
	plt.legend()
	plt.title(rf"For grid of size {size}x{size}")
	plt.savefig(f"./figures/plot_{size}.png", dpi=300)

	best_block_float = block[spdup_float.index(max(spdup_float))]
	best_block_double = block[spdup_double.index(max(spdup_double))]

	print(f"The best block_size for floats for grid of {size}x{size}: {best_block_float} = {int(np.sqrt(best_block_float))}x{int(np.sqrt(best_block_float))}")
	print(f"The best block_size for doubles for grid of {size}x{size}: {best_block_double} = {int(np.sqrt(best_block_double))}x{int(np.sqrt(best_block_double))}")
	print(" ")
