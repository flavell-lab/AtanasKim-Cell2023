function train_decoder(zscored_behaivor, zscored_traces, train, test, beta; metric=cost_mse)
    data_train = zscored_behavior[train]
    data_eval = zscored_behavior[test]
    var_solution = glmnet(transpose(zscored_traces[:,train]), data_train)
    var_predictor = var_solution.betas[:,beta]
    predictions = transpose(zscored_traces) * var_predictor
    eval_metric = metric(predictions[test], data_eval)
    return (var_predictor, predictions, eval_metric)
end


