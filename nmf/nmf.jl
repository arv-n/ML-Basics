### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ b7e702fa-58dc-11eb-22e6-796d3cdcdfb5
begin
	using Plots, PlutoUI,NMF,CSV,DataFrames,LinearAlgebra

		struct TwoColumn{A, B}
			left::A
			right::B
		end
		function Base.show(io, mime::MIME"text/html", tc::TwoColumn)
			write(io,
				"""
				<div style="display: flex;">
					<div style="flex: 50%;">
				""")
			show(io, mime, tc.left)
			write(io,
				"""
					</div>
					<div style="flex: 50%;">
				""")
			show(io, mime, tc.right)
			write(io,
				"""
					</div>
				</div>
			""")
		end
end

# ╔═╡ 9bae7e74-58dc-11eb-00fb-b70e83ab29a0
html"<button onclick='present()'>present</button>"

# ╔═╡ f41136de-58d6-11eb-2503-df3264b933a1
md"""
### Algorithms for Non Negative Matrix Factorization
**Arvind Nayak, January 18, 2021**

based on the paper by *Lee and Seung (2000)*

Seminar **Classical Topics for Machine Learning & Cognitive Algorithms, TU Berlin**
"""

# ╔═╡ 2c55f482-58df-11eb-1abf-47ddfc2c643c
md"""
## Introduction
A *non-negative* matrix (say, $V$) factorized into (usually) two **non-negative** matrices ($W$ and $H$) is known as Non-Negative matrix factorization (NMF).

**What numerical algorithms can be used for learning the optimal factors from data?**
"""

# ╔═╡ 9c3f57ca-58df-11eb-2587-9b8dd03006ac
md"""#### Roadmap 
- Mathematical Description
- An example 
- Cost functions
- Update rules
- Convergence
- Summary
"""

# ╔═╡ 531b4876-58e2-11eb-3f45-37596f16987b
md"""
## Mathematical description
"""

# ╔═╡ 6e0c36b8-58e2-11eb-3b71-fd379b437dc1
md"""
A matrix $V \in \mathbb{R}_{+}^{n,m}$, can be *approximately* factorized into two matrices $W \in \mathbb{R}_{+}^{n,r}$ and $H \in \mathbb{R}_{+}^{r,m}$, where $r = \min(n,m)$ such that, 

$$V \approx WH$$
"""

# ╔═╡ af29ceb2-58e7-11eb-1af8-5ba313594158
md"""This statement can be posed as an optimization problem (*one of them!*).

Find $W \in \mathbb{R}_{+}^{n,r}$, $H \in \mathbb{R}_{+}^{r,m}$, where
```math
\begin{aligned}
& \underset{W,H}{\text{argmin}} 
& & ||V-WH||^{2}_{Fr} \\
& \text{subject to}
& & W \geq 0. \\
& 
& & H \geq 0.
\end{aligned}
```
"""

# ╔═╡ fc91d4c8-5933-11eb-05ce-3d520c9b8db5
md"""
- Convex in either $W$ or $H$, not both. 
- Requires iterative methods *(Q's: Update rules, Initialization)*
"""

# ╔═╡ 4d04a4bc-58ff-11eb-25a0-ab2c800e9a47
md"""
## Mathematical description
"""

# ╔═╡ 693f44fa-58ff-11eb-18ff-0986e2895139
md"""
A matrix $V \in \mathbb{R}_{+}^{n,m}$, can be *approximately* factorized into two matrices $W \in \mathbb{R}_{+}^{n,r}$ and $H \in \mathbb{R}_{+}^{r,m}$, where $r = \min(n,m)$ such that, 

$$V \approx WH$$
"""

# ╔═╡ 413aa29e-5907-11eb-3b7c-b7339ab9d656
n=70;

# ╔═╡ 3e700eaa-5907-11eb-15ed-ef1b380c80ec
m=60;

# ╔═╡ 34d83a7a-5907-11eb-37d5-1150955ef713
V = rand(n,m);

# ╔═╡ 2d73b976-58fd-11eb-2a3b-6d2d5621ef8e
begin
	sl_r = @bind r Slider(1:1:50, default=1);
	md"""
	r = 1 $sl_r 50
	"""
end

# ╔═╡ 3f29aada-58f9-11eb-021b-634ff29cbc74
nmf_alg=nnmf(V,r);

# ╔═╡ 0634bf6e-5906-11eb-0675-1b964d05fd4c
begin
p1 = heatmap(V,
		c=cgrad([:blue, :black, :red]),
		aspect_ratio=1,xlim=(0.25,m+1),ylim=(0,n+1),title="V",legend=false)

p2=heatmap(nmf_alg.W,
		c=cgrad([:blue, :black, :red]),xlim=(0.25,m+1),ylim=(0,n+1),
		aspect_ratio=1,
		axis=false,legend=false,title="W");

p3=heatmap(nmf_alg.H,xlim=(0.25,m+1),ylim=(0,n+1),
		c=cgrad([:blue, :black, :red]),
		aspect_ratio=1,
		axis=false,legend=false,title="H");

plot(p1,p2,p3,layout=(1,3),size=(650,225))
		
end

# ╔═╡ e7c114ce-5907-11eb-34e5-871e45670a34
md"## A more conventional example"

# ╔═╡ 3fe0afec-592c-11eb-3e4f-ef5c46c94fc1
df=CSV.read("mnist_test.csv",DataFrame);

# ╔═╡ 44063e2a-5920-11eb-2e4c-77f9c529e7df
md"""
features (r) = 1 $sl_r 50
"""

# ╔═╡ 1f75b9f2-5932-11eb-2c27-b55defdec452
begin
	sl_show = @bind show_again CheckBox(default=false) 
	md"Show basis and weights $(sl_show)"
end

# ╔═╡ 5f5d91ba-591f-11eb-1aec-c727a59d8318
samples=1000;

# ╔═╡ 281d78d0-5930-11eb-2256-e9894258c40b
md"""
##### NMF done on the MNIST database 

with $(r) features(s) 

Number of examples in the dataset are (m) = $(samples)

Dimensions of the image (n) = 28 x28 = $(28*28)

"""

# ╔═╡ 9b08bafc-592c-11eb-258a-bd5072d490fd
dataset=Matrix(df[1:samples,2:end])'./255;

# ╔═╡ 12788fa6-5912-11eb-1a61-0de03d590799
function solve(dataset,r,algorithm::Symbol,vbose::Bool=false)
	result=nnmf(dataset,r,alg=algorithm,verbose=vbose);
	return result
end

# ╔═╡ 618acd12-592c-11eb-2bf3-9f0602c3b36b
result_mnist_eg = solve(dataset,r,:multmse);

# ╔═╡ 9e13b946-592b-11eb-3662-4f32fb094d08
sl_eg = @bind example Slider(1:1:samples, default=1);

# ╔═╡ 87af0c6c-592d-11eb-26ca-6f84fe46c83f
md"**number** $sl_eg"

# ╔═╡ 1552225c-591a-11eb-16c6-dbee611016bb
TwoColumn(heatmap(rotl90(reshape(dataset[:,example],28,28)),size=(350,300)),
	let
		w=round.(result_mnist_eg.H[:,example],digits=3)
		plot(layout=r,legend=false,size=(350,300))
		img=zeros(r,28,28)
			for i in 1:r
				img[i,:,:]=reshape(result_mnist_eg.W[:,i],28,28);	
				heatmap!(rotl90(img[i,:,:]),clims=(0,1),
				axis=false,aspect_ratio=1,subplot=i,title="$(w[i])")
			end
			current()
	end
	)

# ╔═╡ c89827d2-5931-11eb-3292-432aaf77d4b9
begin
	if show_again
		w=round.(result_mnist_eg.H[:,example],digits=3)
		plot(layout=r,legend=false)
		img=zeros(r,28,28)
		for i in 1:r
			img[i,:,:]=reshape(result_mnist_eg.W[:,i],28,28);	
			heatmap!(rotl90(img[i,:,:]),clims=(0,1),
			axis=false,aspect_ratio=1,subplot=i,title="$(w[i])")
		end
		current()
	end
end

# ╔═╡ c8242768-5933-11eb-1e46-8fe7cb8ff5ca
md"## Cost functions"

# ╔═╡ d7c44dba-5933-11eb-0aaf-03b5b002fb82
md"""
**Problem 1** 
Find $W \in \mathbb{R}_{+}^{n,r}$, $H \in \mathbb{R}_{+}^{r,m}$, where
```math
\begin{aligned}
& \underset{W,H}{\text{argmin}} 
& & ||V-WH||^{2}_{Fr} \\
& \text{subject to}
& & W \geq 0. \\
& 
& & H \geq 0.
\end{aligned}
```

**Problem 2**
Find $W \in \mathbb{R}_{+}^{n,r}$, $H \in \mathbb{R}_{+}^{r,m}$, where
```math
\begin{aligned}
& \underset{W,H}{\text{argmin}} 
& & D(V || WH) \\
& \text{subject to}
& & W \geq 0. \\
& 
& & H \geq 0.
\end{aligned}
```
$D(V||WH) = \underset{ij}\sum\left(V_{ij}\log\frac{V_{ij}}{(WH)_{ij}} - V_{ij} + (WH)_{ij} \right)$

known as the **Kulback-Leibler (KL) divergence** when, $\underset{ij}\sum V_{ij} = \underset{ij}\sum (WH)_{ij} = 1$
"""


# ╔═╡ fc534278-5936-11eb-2b90-2782bc41612d
md"## Update Rules"

# ╔═╡ bb524a9a-593a-11eb-25ea-0d1b89b8d583
md"""
**Algorithm** :  NMF ($V \approx WH$) under Frobenius norm measure.
>**Input**: $V \in \mathbb{R}_{+}^{n,m}$, rank parameter $r \in \mathbb{N}$, Stopping criterion $\epsilon$

>**Output**: $W \in \mathbb{R}_{+}^{n,r}$ and $H \in \mathbb{R}_{+}^{r,m}$

>**Procedure**:
> Define $W^{(0)}$ and $H^{(0)}$ by random or informed initialization. 
>Set $i=0$. Apply the following update rules:\
>$\quad$ (1) Compute: $H^{(i+1)} = H^{(i)} \odot \left(((W^{(i)})^{T}V)/((W^{((i)})^{T}W^{(i)}H^{(i)})\right)$ \
>$\quad$ (2) Compute $W^{(i+1)} = W^{(i)} \odot \left((V(H^{(i+1)})^{T})/((W^{((i)})H^{(i+1)}H^{(i+1)})^{T}\right)$ \
>$\quad$ (3) i = i+1 \
>$\quad$ Repeat the steps (1) to (3) until $||H^{(i)}-H^{(i-1)}|| \leq \epsilon$ and $||W^{(i)}-W^{(i-1)}|| \leq \epsilon$ (or some other stop criterion is fulfilled).

> Set $H = H^{(i)}$ and $W = W^{(i)}$\
>End
"""

# ╔═╡ dbf461ea-593f-11eb-3c4f-3f384cdd0f67
md"## Update Rules"

# ╔═╡ d54bf39c-593f-11eb-18c2-bb28a199e4b4
md"""
**Algorithm** :  NMF ($V \approx WH$) under divergence $D(V||WH)$ measure.
>**Input**: $V \in \mathbb{R}_{+}^{n,m}$, rank parameter $r \in \mathbb{N}$, Stopping criterion $\epsilon$

>**Output**: $W \in \mathbb{R}_{+}^{n,r}$ and $H \in \mathbb{R}_{+}^{r,m}$

>**Procedure**:
> Define $W^{(0)}$ and $H^{(0)}$ by random or informed initialization. 
>Set $i=0$. Apply the following update rules:\
>$\quad$ **(1) Compute: $H^{(i+1)} = H^{(i)} \odot \frac{(W^{(i)})^{T}\frac{V^{(i)}}{W^{(i)}H^{(i)}}}{(W^{(i)})^{T}\mathbf{1}}$ \
>$\quad$ (2) Compute $W^{(i+1)} = W^{(i)} \odot \frac{\frac{V^{(i)}}{W^{(i)}H^{(i+1)}}(H^{(i+1)})^{T}}{\mathbf{1}(H^{(i+1)})^{T}}$** \
>$\quad$ (3) i = i+1 \
>$\quad$ Repeat the steps (1) to (3) until $||H^{(i)}-H^{(i-1)}|| \leq \epsilon$ and $||W^{(i)}-W^{(i-1)}|| \leq \epsilon$ (or some other stop criterion is fulfilled).

> Set $H = H^{(i)}$ and $W = W^{(i)}$\
>End
"""

# ╔═╡ 17395672-5946-11eb-04d7-bbed3f52765a
md"## A note on Initialization"

# ╔═╡ 618e818c-596b-11eb-2efa-b3bd04b8f7c3
md"""
- Since we search for a local minima, initialization does affect our final approximation

- Most common method is **Random initialization**

- Many improvements have been suggested such as SVD based initialization. See [^3]
"""

# ╔═╡ fc290008-5945-11eb-0401-53f35b7b7430
md"## Convergence Results"

# ╔═╡ e0f6ebf0-5946-11eb-061a-f77167cc1202
md""" 
**features** (r) = 1 $sl_r 50
"""

# ╔═╡ 9a520c6c-5963-11eb-3944-39607044002f
dataset_conv_eg = [0. 1 3 4 5 6 7;
				   0 1 3 3 2 1 0;
				   0 0 0.3 0 0 0.2 0;
				   7 0 0 0.1 0.1 0.1 0;
				   7 6 5 4 3 2 1]
	#rand(0:0.01:1,10,10); 

# ╔═╡ 8cd23380-5947-11eb-23dd-0dc67c662d3a
W, H = NMF.randinit(dataset_conv_eg, r);

# ╔═╡ 8d0d33fe-5947-11eb-1ae1-515e256e5065
TwoColumn(
	heatmap(rotl90(dataset_conv_eg),size=(325,325),title="Target",axis=false),
	heatmap(rotl90(W*H),size=(325,325),axis=false,legend=false,
		title="WH L2_norm = $(round(norm(dataset_conv_eg-(W*H),2),digits=3))"
	)
	)	

# ╔═╡ 7215d93e-5974-11eb-1a9e-5303645979bd
md"## Convergence Results"

# ╔═╡ 6b8c5630-5954-11eb-0706-9d77076e7cb9
function solve_nmf(data::Array{Float64},W::Array{Float64},
		H::Array{Float64},alg,maxiters=100,tol=1.e-5)
	
	eps_mc = eps(Float64)
	
	n = size(data,1);
	m = size(data,2);
	r = size(W,2);
	
	H_W_error = zeros(2,maxiters);
	L2error = zeros(maxiters);
	ell = 1
	below_thresh = false
	
	if alg=="mse"
		while !(below_thresh) && (ell <= maxiters)
			H_ell = H
			W_ell = W
			print(size(W))
			#Update
			H .* ((W'*data) ./ ((W'*W)*H) .+ eps_mc) 
			W = W .* ((data*H') ./ ((W*H)*H') .+ eps_mc)

			#error
			H_err = norm(H-H_ell,2)
			W_err = norm(W-W_ell,2)
			L2error[ell] = norm(data-W*H,2)

			H_W_error[:,ell] = hcat(H_err,W_err)

			if H_err < tol && W_err < tol
				below_thresh = true
				H_W_error = H_W_error[:,0:ell]
				L2error = L2error[0:ell]
			end
			ell += 1
		end
	end
	
	return W, H, H_W_error,L2error
end

# ╔═╡ 8c7497f2-5947-11eb-296c-edad73197092
W_eg, H_eg, HW_error,L2error = solve_nmf(dataset_conv_eg,W,H,"mse");

# ╔═╡ 8c55543c-5947-11eb-2f5a-a92a31aff791
TwoColumn(
	heatmap(rotl90(dataset_conv_eg),size=(325,325),
		title="Target",legend=false,axis=false),
	heatmap(rotl90(W_eg*H_eg),size=(325,325),legend=false,axis=false,
		title="WH L2_norm = $(round(norm(dataset_conv_eg-(W_eg*H_eg),2),digits=3))"
	)
	)

# ╔═╡ 62977ee6-5966-11eb-3ed9-13b85e5fd307
md"## Convergence "

# ╔═╡ 2a9a6272-5966-11eb-1e6f-ab707dceedb8
plot(L2error,yaxis=:log,ylabel="L2 Norm",xlabel ="Iteration Index",legend=false)

# ╔═╡ bc09a9fe-5969-11eb-24ec-2fb61a074779
md"## Convergence "

# ╔═╡ f1c5929c-5969-11eb-0d51-b1ea82b7b4e8
md"""
**example** $sl_eg 
"""

# ╔═╡ 6ccde5f8-5969-11eb-1618-07d91325cfbb
eg2 = reshape(dataset[:,example],28,28);

# ╔═╡ d055db26-5969-11eb-01a0-796bc564470d
W2, H2 = NMF.randinit(eg2, r);

# ╔═╡ 13fbf23e-596a-11eb-30a3-4b8b8a30dcd5
TwoColumn(
	heatmap(rotl90(eg2),size=(325,325),title="Target",axis=false),
	heatmap(rotl90(W2*H2),size=(325,325),axis=false,legend=false,
		title="WH L2_norm = $(round(norm(eg2-(W2*H2),2),digits=3))"
	)
	)

# ╔═╡ be4d2a96-596a-11eb-34ff-6de9f113c2cb
md"## Convergence"

# ╔═╡ d32e90da-596a-11eb-368d-674c97edbfaf
md"**features** (r) = 1 $sl_r 50"

# ╔═╡ d03054ae-5974-11eb-2d74-e11f6bf1d52e
result_conv_eg = NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse,maxiter=100), eg2, W2, H2);

# ╔═╡ a52c8296-5974-11eb-0d58-e30ad20f338a
TwoColumn(
	heatmap(rotl90(eg2),size=(325,325),title="Target",axis=false),
	heatmap(rotl90(result_conv_eg.W*result_conv_eg.H),size=(325,325),axis=false,legend=false,
		title="WH L2_norm = $(round(norm(eg2-(result_conv_eg.W*result_conv_eg.H),2),digits=3))"
	)
	)

# ╔═╡ ae39643a-5974-11eb-045e-f9b1f2111ae9
md"## Convergence"

# ╔═╡ d8cb379e-596c-11eb-13de-43bb7692d57b
N = 6;

# ╔═╡ bd3df5d4-596c-11eb-363f-35dccd4616b5
data_eg = rand([0.,1.,10],N,N);

# ╔═╡ 84dbc798-596c-11eb-154e-ab85c3f3ab4a
W3, H3 = NMF.randinit(data_eg, r);

# ╔═╡ 99e02b5c-596c-11eb-340a-f5d9b56d111c
function compare(data,W,H)
	mse = NMF.solve!(NMF.MultUpdate{Float64}(obj=:mse,maxiter=1000), data, W3, H3);
	kl = NMF.solve!(NMF.MultUpdate{Float64}(obj=:div,maxiter=1000), data, W3, H3);
	mse_iters = mse.niters;
	kl_iters = kl.niters;
	return [mse_iters kl_iters];
end

# ╔═╡ 4c5fc62a-596d-11eb-00a1-13898287fa1d
iters = compare(data_eg,W3,H3);

# ╔═╡ d5931bfe-596d-11eb-24ed-5b206abba970
begin
	x = ["MSE", "Divergence"]
	bar(x,iters',label="max iters",bar_width = 0.25,framestyle = :box, grid = false, yticks = 0:maximum(iters)/10:maximum(iters))
end

# ╔═╡ 4624578c-5970-11eb-08c0-a101466734bc
md"""# Summary and Current work 

- widely used tool for the analysis of high-dimensional data

- Other algorithms: other divergences (chi-square statistic), Pearson-Neyman distances, etc 

- ALS Projected Gradient Methods, Coordinate Descent Methods

- searching for global minima of the factors and factor initialization.

- how to factorize million-by-billion matrices, (Distributed Nonnegative Matrix Factorization) Scalable Nonnegative Matrix Factorization (ScalableNMF) etc.
"""


# ╔═╡ b5187bda-5972-11eb-0d29-93484f0f91c1
md"# References
[^1]: Lee, DD & Seung, HS (1999). Learning the parts of objects by non-negative matrix factorization. Nature.

[^2]: D. D. Lee and H. S. Seung.  Algorithms for non-negative matrixfactorization. InNeural Information Processing Systems, pages 556–562, 2000.

[^3]: C. Boutsidis and E. Gallopoulos. SVD based initialization: a headstart for nonnegative matrix factorization.Pattern Recognition,41(4):1350–1362, 2008.

[^4]: S.P. Boyd and L. Vandenberghe.Convex optimization. CambridgeUniversity Press, 2004

[^5]: Pluto.jl, Julia, https://juliahub.com/docs/Pluto/OJqMt/0.7.4/ 

[^6]: LeCun, Y. & Cortes, C. (2010). MNIST handwritten digit database
"

# ╔═╡ Cell order:
# ╟─9bae7e74-58dc-11eb-00fb-b70e83ab29a0
# ╟─b7e702fa-58dc-11eb-22e6-796d3cdcdfb5
# ╠═f41136de-58d6-11eb-2503-df3264b933a1
# ╟─2c55f482-58df-11eb-1abf-47ddfc2c643c
# ╟─9c3f57ca-58df-11eb-2587-9b8dd03006ac
# ╟─531b4876-58e2-11eb-3f45-37596f16987b
# ╟─6e0c36b8-58e2-11eb-3b71-fd379b437dc1
# ╠═af29ceb2-58e7-11eb-1af8-5ba313594158
# ╟─fc91d4c8-5933-11eb-05ce-3d520c9b8db5
# ╟─4d04a4bc-58ff-11eb-25a0-ab2c800e9a47
# ╟─693f44fa-58ff-11eb-18ff-0986e2895139
# ╠═413aa29e-5907-11eb-3b7c-b7339ab9d656
# ╠═3e700eaa-5907-11eb-15ed-ef1b380c80ec
# ╠═34d83a7a-5907-11eb-37d5-1150955ef713
# ╟─0634bf6e-5906-11eb-0675-1b964d05fd4c
# ╠═2d73b976-58fd-11eb-2a3b-6d2d5621ef8e
# ╟─3f29aada-58f9-11eb-021b-634ff29cbc74
# ╟─e7c114ce-5907-11eb-34e5-871e45670a34
# ╟─281d78d0-5930-11eb-2256-e9894258c40b
# ╠═3fe0afec-592c-11eb-3e4f-ef5c46c94fc1
# ╠═618acd12-592c-11eb-2bf3-9f0602c3b36b
# ╟─87af0c6c-592d-11eb-26ca-6f84fe46c83f
# ╟─44063e2a-5920-11eb-2e4c-77f9c529e7df
# ╟─1552225c-591a-11eb-16c6-dbee611016bb
# ╟─1f75b9f2-5932-11eb-2c27-b55defdec452
# ╟─c89827d2-5931-11eb-3292-432aaf77d4b9
# ╠═5f5d91ba-591f-11eb-1aec-c727a59d8318
# ╠═9b08bafc-592c-11eb-258a-bd5072d490fd
# ╟─12788fa6-5912-11eb-1a61-0de03d590799
# ╟─9e13b946-592b-11eb-3662-4f32fb094d08
# ╟─c8242768-5933-11eb-1e46-8fe7cb8ff5ca
# ╟─d7c44dba-5933-11eb-0aaf-03b5b002fb82
# ╟─fc534278-5936-11eb-2b90-2782bc41612d
# ╟─bb524a9a-593a-11eb-25ea-0d1b89b8d583
# ╟─dbf461ea-593f-11eb-3c4f-3f384cdd0f67
# ╟─d54bf39c-593f-11eb-18c2-bb28a199e4b4
# ╟─17395672-5946-11eb-04d7-bbed3f52765a
# ╟─618e818c-596b-11eb-2efa-b3bd04b8f7c3
# ╟─fc290008-5945-11eb-0401-53f35b7b7430
# ╟─e0f6ebf0-5946-11eb-061a-f77167cc1202
# ╠═9a520c6c-5963-11eb-3944-39607044002f
# ╠═8cd23380-5947-11eb-23dd-0dc67c662d3a
# ╟─8d0d33fe-5947-11eb-1ae1-515e256e5065
# ╠═8c7497f2-5947-11eb-296c-edad73197092
# ╟─7215d93e-5974-11eb-1a9e-5303645979bd
# ╟─8c55543c-5947-11eb-2f5a-a92a31aff791
# ╟─6b8c5630-5954-11eb-0706-9d77076e7cb9
# ╟─62977ee6-5966-11eb-3ed9-13b85e5fd307
# ╟─2a9a6272-5966-11eb-1e6f-ab707dceedb8
# ╟─bc09a9fe-5969-11eb-24ec-2fb61a074779
# ╟─f1c5929c-5969-11eb-0d51-b1ea82b7b4e8
# ╠═6ccde5f8-5969-11eb-1618-07d91325cfbb
# ╠═d055db26-5969-11eb-01a0-796bc564470d
# ╟─13fbf23e-596a-11eb-30a3-4b8b8a30dcd5
# ╟─be4d2a96-596a-11eb-34ff-6de9f113c2cb
# ╟─d32e90da-596a-11eb-368d-674c97edbfaf
# ╠═d03054ae-5974-11eb-2d74-e11f6bf1d52e
# ╟─a52c8296-5974-11eb-0d58-e30ad20f338a
# ╟─ae39643a-5974-11eb-045e-f9b1f2111ae9
# ╠═d8cb379e-596c-11eb-13de-43bb7692d57b
# ╟─99e02b5c-596c-11eb-340a-f5d9b56d111c
# ╟─d5931bfe-596d-11eb-24ed-5b206abba970
# ╟─bd3df5d4-596c-11eb-363f-35dccd4616b5
# ╟─84dbc798-596c-11eb-154e-ab85c3f3ab4a
# ╟─4c5fc62a-596d-11eb-00a1-13898287fa1d
# ╟─4624578c-5970-11eb-08c0-a101466734bc
# ╟─b5187bda-5972-11eb-0d29-93484f0f91c1
