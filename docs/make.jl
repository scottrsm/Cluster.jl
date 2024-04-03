using Cluster
import Pkg

Pkg.add("Documenter")
using Documenter

makedocs(
	sitename = "Cluster",
	format = Documenter.HTML(),
	modules = [Cluster]
	)

	# Documenter can also automatically deploy documentation to gh-pages.
	# See "Hosting Documentation" and deploydocs() in the Documenter manual
	# for more information.
	deploydocs(
		repo = "github.com/scottrsm/Cluster.jl.git"
	)
