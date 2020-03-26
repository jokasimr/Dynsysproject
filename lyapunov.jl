using LinearAlgebra, GenericLinearAlgebra, StaticArrays, Statistics


iter(f, x, k) = (for i=1:k x=f(x) end; x)

function attractor_points(f, start, n; bound=5, iters=100)
    inside_bound(p) = sum(p.^2) < bound^2
    new_points(k) = [start + bound/2 * randn(length(start)) for _=1:k]
    x = []
    while length(x) < n
        new = iter.(f, new_points(n), iters)
        x = vcat(x, new[inside_bound.(new)])
    end
    x[1:n]
end


henon(x; a=1.4, b=0.3) = ((x, y) = x; SVector(1 - a*x^2 + y, b*x))
henonjac(x; a=1.4, b=0.3) = @SMatrix [-2a*x[1] 1; b 0]

qrlyap = function(x, k)
    DT = SMatrix{2,2}(1I)
    ly = zeros(2)
    for i=1:k
        F = qr(henonjac(x)*DT)
        DT = F.Q
        ly .+= log.(abs.(diag(F.R)))
        x = henon(x)
    end
    ly ./ k
end
