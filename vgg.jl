using Flux
using Flux: onehotbatch, onecold, flatten
using Flux.Losses: mae
using Flux.Data: DataLoader
using Parameters: @with_kw
using Statistics: mean
using CUDA
using MLDatasets: CIFAR10
using MLUtils: splitobs
using LinearAlgebra: norm

if CUDA.has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

function get_processed_data(args)
    x, y = CIFAR10(:train)[:]

    # Training and validation
    (train_x, train_y), (val_x, val_y) = splitobs((x, y), at=1-args.valsplit)

    train_x = float(train_x)
    train_y = onehotbatch(train_y, 0:9)
    val_x = float(val_x)
    val_y = onehotbatch(val_y, 0:9)
    
    return (train_x, train_y), (val_x, val_y)
end

function get_test_data()
    test_x, test_y = CIFAR10(:test)[:]
   
    test_x = float(test_x)
    test_y = onehotbatch(test_y, 0:9)
    
    return test_x, test_y
end

function normalize!(xs)
    for i in 1:size(xs)[2]
        xs[:,i] .= xs[:,i] / norm(xs[:,i])
    end
    xs
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

function vgg_direction()
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
        normalize!
    ])
end

@with_kw mutable struct Args
    batchsize::Int = 128
    lr::Float64 = 5e-5
    epochs::Int = 50
    valsplit::Float64 = 0.1
end

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)
	
    # Load the train, validation data 
    train_data, val_data = get_processed_data(args)
    
    train_loader = DataLoader(train_data, batchsize=args.batchsize, shuffle=true)
    val_loader = DataLoader(val_data, batchsize=args.batchsize)

    @info("Constructing Model")	
    m = vgg_direction() |> gpu

    loss(x, y) = mae(m(x), y)

    ## Training
    # Defining the optimizer
    opt = ADAM(args.lr)
    ps = Flux.params(m)

    @info("Training....")
    # Starting to train models
    for epoch in 1:args.epochs
        @info "Epoch $epoch"

        for (x, y) in train_loader
            x, y = x |> gpu, y |> gpu
            gs = Flux.gradient(() -> loss(x,y), ps)
            Flux.update!(opt, ps, gs)
        end

        validation_loss = 0f0
        for (x, y) in val_loader
            x, y = x |> gpu, y |> gpu
            validation_loss += loss(x, y)
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
