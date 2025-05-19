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

	df_cpu = df[ (df[n]==size) & (df[cpu]==0)]
	df_gpu = df[ (df[n]==size) & (df[gpu]==1) ]

	print(df_cpu)

	t_float_cpu = (df_cpu[time_cpu_float])
	t_float_gpu = np.array(df_gpu[time_gpu_float])
	t_double_cpu = np.array(df_cpu[time_cpu_double])
	t_double_gpu = np.array(df_gpu[time_gpu_double])

	print(t_float_cpu)
	spdup_float = t_float_cpu / t_float_gpu
	spdup_double = t_double_cpu / t_double_gpu
		
	block = df_gpu[block_sizes]
	print(block)
	print(spdup_float)

	plt.plot(block, spdup_float, label=f"float")
	plt.plot(block, spdup_double, label=f"double")

	plt.xlabel("block size")
	plt.ylabel("speedup")
	plt.legend()
	plt.title(f"")
	plt.savefig(f"./figures/plot_{size}.png", dpi=300)
