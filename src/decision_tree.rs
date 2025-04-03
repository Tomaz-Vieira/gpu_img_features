use anyhow as ah;

use graphviz_rust as gv;
use graphviz_rust::dot_structures as gs;

pub struct DecisionTree{
    stmts: Vec<gs::Stmt>,
}

impl DecisionTree{
    pub fn parse(dot: &str) -> ah::Result<Self>{
        let graph: gs::Graph = gv::parse(dot)
            .map_err(|s| ah::anyhow!("Could not parse the dot syntax: {s}"))?;
        let gs::Graph::DiGraph { stmts,.. } = graph else {
            ah::bail!("Expected directed graph");
        };
        Ok(Self{stmts})
    }

    pub fn write_wgsl(&self, out: &mut impl std::fmt::Write){
        for stmt in &self.stmts{
            dbg!(stmt);
        }
    }
}


#[test]
fn test_decision_tree_parsing(){
    let dt = DecisionTree::parse(r#"
        digraph Tree {
            node [shape=box, fontname="helvetica"] ;
            edge [fontname="helvetica"] ;
            0 [label="x[16] <= 36.255\ngini = 0.5\nsamples = 88\nvalue = [68, 71]"] ;
            1 [label="x[17] <= 0.954\ngini = 0.283\nsamples = 51\nvalue = [68.0, 14.0]"] ;
            0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
            2 [label="gini = 0.0\nsamples = 45\nvalue = [68, 0]"] ;
            1 -> 2 ;
            3 [label="gini = 0.0\nsamples = 6\nvalue = [0, 14]"] ;
            1 -> 3 ;
            4 [label="gini = 0.0\nsamples = 37\nvalue = [0, 57]"] ;
            0 -> 4 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
        }
    "#).unwrap();

    let mut out = String::new();
    dt.write_wgsl(&mut out);
}
