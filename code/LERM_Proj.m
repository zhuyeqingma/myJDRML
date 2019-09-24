function [projection_x, projection_y] = LERM_Proj(kernel_x,kernel_y,model,dim,V)
mux = model.mux;
sigx = model.sigx;
muy = model.muy;
sigy = model.sigy;
kernel_x_z = ZeroMeanOneVar(kernel_x',mux',sigx');
kernel_y_z = ZeroMeanOneVar(kernel_y',muy',sigy');
projection_x=(model.eigx(:,1:dim))'*(kernel_x_z'*model.eigx_p)';
projection_y =(model.eigy(:,1:dim))'*(kernel_y_z'*model.eigy_p)';




