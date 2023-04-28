use super::BoxOp;


#[derive(Debug)]
pub struct OpGraph {
    args: Vec<Option<OpGraph>>,
    //tensor: Option<Tensor<D>>,
    op: BoxOp,
}

impl OpGraph {
    pub fn new(args: &[&Option<OpGraph>], op: BoxOp) -> OpGraph {
        Self {
            args: args.iter().map(|g| 
                if let Some(graph) = g {
                    Some(graph.clone())
                } else {
                    None
                }
            ).collect(),
            op,
        }
    }
}

impl Clone for OpGraph {
    fn clone(&self) -> Self {
        Self { 
            args: self.args.clone(), 
            op: self.op.box_clone()
         }
    }
}
