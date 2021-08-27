### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ b415892a-9087-11eb-0cca-b7b3301a31c3
begin
	using PlutoUI
	using Lasso,CSV,DataFrames
end

# ╔═╡ fa43be4e-9087-11eb-3782-95121a27259c
df=CSV.read("psa.csv",DataFrame)

# ╔═╡ a9e1fb9c-90d6-11eb-0722-5713b4be0a94
size(df)

# ╔═╡ Cell order:
# ╠═b415892a-9087-11eb-0cca-b7b3301a31c3
# ╠═fa43be4e-9087-11eb-3782-95121a27259c
# ╠═a9e1fb9c-90d6-11eb-0722-5713b4be0a94
