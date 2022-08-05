using Flux
using Flux.Losses: mae
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using MLUtils: splitobs
using LinearAlgebra: norm
using HDF5
using Random: shuffle

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function get_energy_data(args)
    f = h5open("$(@__DIR__)/../data/data.h5", "r")
    pulses = permutedims(f["pulses"][:, :, 1:3000], (2, 1, 3))
    x = ones(size(pulses)[1], size(pulses)[2], 1, size(pulses)[3])
    x[:,:,1,:] = pulses
    energies = log.(10, f["energies"][1:3000])
    y = zeros(1, size(energies)[1])
    y[:,:] = energies
    x, y = x .|> Float32, y .|> Float32

    # Training and validation
    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = float(train_y)
    val_x = float(val_x)
    val_y = float(val_y)
    return (train_x, train_y), (val_x, val_y)
end

function get_direction_data(args)
    f = h5open("$(@__DIR__)/../data/data.h5", "r")
    pulses = permutedims(f["pulses"][:,:,1:3000], (2, 1, 3))
    x = ones(size(pulses)[1], size(pulses)[2], 1, size(pulses)[3])
    x[:,:,1,:] = pulses
    y = f["directions"][:,1:3000]
    x, y = x .|> Float32, y .|> Float32

    # Training and validation
    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = float(train_y)
    val_x = float(val_x)
    val_y = float(val_y)
    return (train_x, train_y), (val_x, val_y)
end

function get_test_energy_data()
    f = h5open("$(@__DIR__)/../data/data.h5", "r")
    pulses = permutedims(f["pulses"][:,:,3000:end], (2, 1, 3))
    x = ones(size(pulses)[1], size(pulses)[2], 1, size(pulses)[3])
    x[:,:,1,:] = pulses
    energies = log.(10, f["energies"][3000:end])
    y = zeros(1, size(energies)[1])
    y[:,:] = energies
    x, y = x .|> Float32, y .|> Float32

    # Training and validation
    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = float(train_y)
    val_x = float(val_x)
    val_y = float(val_y)
    return (train_x, train_y), (val_x, val_y)
end

function get_test_directions_data()
    f = h5open("$(@__DIR__)/../data/data.h5", "r")
    pulses = permutedims(f["pulses"][:,:,3000:end], (2, 1, 3))
    x = ones(size(pulses)[1], size(pulses)[2], 1, size(pulses)[3])
    x[:,:,1,:] = pulses
    y = f["directions"][:,3000:end]
    x, y = x .|> Float32, y .|> Float32

    # Training and validation
    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = float(train_y)
    val_x = float(val_x)
    val_y = float(val_y)
    return (train_x, train_y), (val_x, val_y)
end

function normalize!(xs)
    for i in 1:size(xs)[2]
        xs[:,i] .= xs[:,i] / norm(xs[:,i])
    end
    xs
end

function vgg_direction()
    println("cock")
    Chain([
        Conv((1,5), 1 => 32, pad = (0,0,2,2)),
        Conv((1,5), 32 => 32, pad = (0,0,2,2)),
        Conv((1,5), 32 => 32, pad = (0,0,2,2)),
        MeanPool((1,4)),
        Conv((1,5), 32 => 64, pad = (0,0,2,2)),
        Conv((1,5), 64 => 64, pad = (0,0,2,2)),
        Conv((1,5), 64 => 64, pad = (0,0,2,2)),
        MeanPool((1,4)),
        Conv((1,5), 64 => 128, pad = (0,0,2,2)),
        Conv((1,5), 128 => 128, pad = (0,0,2,2)),
        Conv((1,5), 128 => 128, pad = (0,0,2,2)),
        MeanPool((1,4)),
        Conv((1,5), 128 => 256, pad = (0,0,2,2)),
        Conv((1,5), 256 => 256, pad = (0,0,2,2)),
        Conv((1,5), 256 => 256, pad = (0,0,2,2)),
        MeanPool((1,4)),
        BatchNorm(256),
        Flux.flatten,
        Dense(2560, 1024),
        Dense(1024, 1024),
        Dense(1024, 512),
        Dense(512, 256),
        Dense(256, 128),
        Dense(128, 3),
    ])
end

function vgg_energy()
    Chain([
        Conv((1,5), 1 => 32, pad = (0,0,2,2)),
        Conv((1,5), 32 => 32, pad = (0,0,2,2)),
        Conv((1,5), 32 => 32, pad = (0,0,2,2)),
        MeanPool((1,4)),
        Conv((1,5), 32 => 64, pad = (0,0,2,2)),
        Conv((1,5), 64 => 64, pad = (0,0,2,2)),
        Conv((1,5), 64 => 64, pad = (0,0,2,2)),
        MeanPool((1,4)),
        Conv((1,5), 64 => 128, pad = (0,0,2,2)),
        Conv((1,5), 128 => 128, pad = (0,0,2,2)),
        Conv((1,5), 128 => 128, pad = (0,0,2,2)),
        MeanPool((1,4)),
        Conv((1,5), 128 => 256, pad = (0,0,2,2)),
        Conv((1,5), 256 => 256, pad = (0,0,2,2)),
        Conv((1,5), 256 => 256, pad = (0,0,2,2)),
        MeanPool((1,4)),
        BatchNorm(256),
        Flux.flatten,
        Dense(2560, 1024),
        Dense(1024, 1024),
        Dense(1024, 512),
        Dense(512, 256),
        Dense(256, 128),
        Dense(128, 1),
    ])
end

function mine(joint, marginal)
    (mean(joint, dims=2) - log.(mean(exp.(marginal[isfinite.(exp.(marginal))]), dims=1)))[1,1]
end

function mine_loss(mx, y, yshuffle)
    joint = abs.(mx .- y)
    marginal = abs.(mx .- yshuffle)
    -mine(joint, marginal)
end

@with_kw mutable struct Args
    batchsize::Int = 128
    lr::Float64 = 5e-5
    epochs::Int = 50
    valsplit::Float64 = 0.1
    loss::Function = mine_loss
end

function train_energy(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train_data, val_data = get_energy_data(args)
    
    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")	
    m = vgg_energy() |> gpu

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:args.epochs
        @info "Epoch $epoch"

        for (x, y) in train_loader
            x, y, yshuffle = x |> gpu, y |> gpu, shuffle(y) |> gpu

            gs = Flux.gradient(() -> args.loss(m(x), y, yshuffle), ps)
            Flux.update!(opt, ps, gs)
        end

        # This is all validation. Shouldn't matter
        validation_loss = 0f0
        for (x, y) in val_loader
            x, y= x |> gpu, y |> gpu
            validation_loss += mae(m(x), y)
        end
        validation_loss /= length(val_loader)
        @show validation_loss
    end

    return m
end

function test(m; kws...)
    args = Args(kws...)

    test_data = get_test_data()
    test_loader = DataLoader(test_data, batchsize=args.batchsize)

    correct, total = 0, 0
    for (x, y) in test_loader
        x, y = x |> gpu, y |> gpu
        correct += sum(onecold(cpu(m(x))) .== onecold(cpu(y)))
        total += size(y, 2)
    end
    test_accuracy = correct / total

    # Print the final accuracy
    @show test_accuracy
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = train()
    test(m)
end
