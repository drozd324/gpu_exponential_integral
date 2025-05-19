#align(center, text(17pt)[
	*CUDA HW3 | Exponential Integral*
	])

#align(center)[
	Patryk Drozd\
	Trinity College Dublin\
	#link("mailto:drozdp@tcd.ie")
]

= Task 1

Output from plotting.py

#raw("
	The best block_size for floats for grid of 5000x5000: 361
	The best block_size for doubles for grid of 5000x5000: 256
	 
	The best block_size for floats for grid of 8192x8192: 900
	The best block_size for doubles for grid of 8192x8192: 784
	 
	The best block_size for floats for grid of 16384x16384: 256
	The best block_size for doubles for grid of 16384x16384: 256
	 
	The best block_size for floats for grid of 20000x20000: 900
	The best block_size for doubles for grid of 20000x20000: 1024
")

#grid(
	columns: (auto, auto),
	rows: (auto, auto),

	figure(
		image("./GPUexponentialIntegral/figures/plot_5000.png", width: 100%),
	),
	figure(
		image("./GPUexponentialIntegral/figures/plot_8192.png", width: 100%),
	),

	figure(
		image("./GPUexponentialIntegral/figures/plot_16384.png", width: 100%),
	),
	figure(
		image("./GPUexponentialIntegral/figures/plot_20000.png", width: 100%),
	)
)


== Task 2


