use std::collections::HashMap;

use anyhow::{self as ah, Context};

use graphviz_rust as gv;
use graphviz_rust::dot_structures as gs;

#[derive(Debug, Copy, Clone)]
struct Decision{
    feature_idx: usize,
    threshold: f32,
}

impl Decision{
    fn try_parse_label_attr(s: &str) -> ah::Result<Option<Self>> {
        let Some(suffix) = s.strip_prefix("x[") else {
            return Ok(None)
        };
        let Some((feature_idx_raw, suffix)) = suffix.split_once("]") else {
            ah::bail!("Could not find feature index")
        };
        let feature_idx = feature_idx_raw.parse::<usize>().context("Parsing feature index")?;
        let Some((prefix, threshold_raw)) = suffix.split_once(" <= ") else {
            ah::bail!("Could not find threshold");
        };
        if !prefix.is_empty(){
            ah::bail!("Could not find threshold");
        }
        let threshold = threshold_raw.parse::<f32>().context("Parsing threshold")?;
        Ok(Some(Self{feature_idx, threshold}))
    }
}

#[derive(Debug, Copy, Clone)]
struct Prediction{
    class: usize,
}

impl Prediction{
    fn try_parse_label_attr(s: &str) -> ah::Result<Option<Self>> {
        let Some(class_raw) = s.strip_prefix("class = ") else {
            return Ok(None)
        };
        let class = class_raw.parse::<usize>().context(format!("Parsing class from {s} >>{class_raw}<<"))?;
        Ok(Some(Self{class}))
    }
}

///////////////////////

#[derive(Debug, Copy, Clone)]
struct Edge{
    origin: u32,
    target: u32,
}

#[derive(Debug)]
enum DecisionNode{
    Decision{
        decision: Decision,
        le_child: Box<DecisionNode>,
        gt_child: Box<DecisionNode>,
    },
    Prediction(Prediction),
}

fn write_indent(writer: &mut impl std::fmt::Write, level: usize) -> Result<(), std::fmt::Error>{
    for _ in 0..level{
        write!(writer, "    ")?;
    }
    Ok(())
}

impl DecisionNode{
    fn write_wgsl(&self, code: &mut impl std::fmt::Write, indent_level: usize) -> Result<(), std::fmt::Error>{
        match self{
            Self::Prediction(pred) => {
                write_indent(code, indent_level)?;
                write!(code, "classs_{}_score += 1;\n", pred.class)
            },
            Self::Decision { decision, le_child, gt_child } => {
                let Decision { feature_idx, threshold } = decision;

                write_indent(code, indent_level)?;
                let feature_var_idx = feature_idx / 3; //FIXME: assuming image is RGB
                let feature_component_idx = feature_idx % 3; //FIXME: assuming image is RGB
                write!(code, "if feature_{feature_var_idx}[{feature_component_idx}] <= {threshold} {{\n")?;
                    le_child.write_wgsl(code, indent_level + 1)?;
                write_indent(code, indent_level)?;
                write!(code, "}} else {{\n")?;
                    gt_child.write_wgsl(code, indent_level + 1)?;
                write_indent(code, indent_level)?;
                write!(code, "}}\n")?;
                Ok(())
            }
        }
    }
}


fn parse_vert(vert: &gs::Vertex) -> ah::Result<u32>{
    let gs::Vertex::N(node) = vert else{
        return Err(ah::anyhow!("Vert is not a node: {vert:?}"))
    }; 
    let node_id: &gs::Id = &node.0;
    node_id.to_string().parse::<u32>().context("Parsing vertex id")
}

fn parse_edge(edge: &gs::Edge) -> ah::Result<Edge>{
    let gs::EdgeTy::Pair(v1, v2) = &edge.ty else {
        return Err(ah::anyhow!("Don't know how to handle non-pair edges"))
    };
    Ok(Edge{
        origin: parse_vert(v1)?,
        target: parse_vert(v2)?,
    })
}

pub struct DecisionTree{
    #[allow(dead_code)]
    root: DecisionNode,
}

impl DecisionTree{
    pub fn parse(dot: &str) -> ah::Result<Self>{
        let graph: gs::Graph = gv::parse(dot)
            .map_err(|s| ah::anyhow!("Could not parse the dot syntax: {s}"))?;
        let gs::Graph::DiGraph { stmts,.. } = graph else {
            ah::bail!("Expected directed graph");
        };

        let edges: Vec<_> = stmts.iter()
            .filter_map(|s| match s{
                gs::Stmt::Edge(edge) => Some(edge),
                _ => None
            })
            .map(parse_edge)
            .collect::<ah::Result<_>>()?;

        let mut predictions = HashMap::<u32, Prediction>::new();
        let mut decisions = HashMap::<u32, Decision>::new();
        'stmts: for s in &stmts{
            let gs::Stmt::Node(node) = s else {
                continue;
            };
            let node_id = node.id.0.to_string().parse::<u32>().context("Parsing node Id of {node:?}")?;
            dbg!(node_id);
            let is_leaf = edges.iter().find(|edge| edge.origin == node_id).is_none();
            let Some(label_attr) = node.attributes.iter().find(|attr| attr.0.to_string() == "label") else {
                ah::bail!("Node has no label {node:?}");
            };
            let raw_label = label_attr.1.to_string();
            let trimmed_label = raw_label.trim_matches('"');
            for label_attr in trimmed_label.split("\\n"){
                if is_leaf{
                    let Some(prediction) = Prediction::try_parse_label_attr(&label_attr)? else {
                        continue
                    };
                    predictions.insert(node_id, prediction);
                    continue 'stmts
                } else {
                    let Some(decision) = Decision::try_parse_label_attr(&label_attr)? else {
                        continue
                    };
                    decisions.insert(node_id, decision);
                    continue 'stmts
                }
            }
            ah::bail!("Could not parse statement {s:?}")
        }

        dbg!(&predictions);
        dbg!(&decisions);

        fn build_tree(
            node_id: u32,
            edges: &[Edge],
            predictions: &HashMap<u32, Prediction>,
            decisions: &HashMap<u32, Decision>,
        ) -> ah::Result<DecisionNode>{
            if let Some(pred) = predictions.get(&node_id){
                return Ok(DecisionNode::Prediction(*pred))
            }
            if let Some(dec) = decisions.get(&node_id){
                //FIXME: assume first edge is the "True" one. Is this reliable?
                let out_edges: [Edge; 2] = edges.iter()
                    .filter(|edge| edge.origin == node_id)
                    .cloned()
                    .collect::<Vec<Edge>>()
                    .try_into()
                    .map_err(|_| ah::anyhow!("Expected two edges from {node_id}"))?;
                return Ok(DecisionNode::Decision{
                    decision: *dec,
                    le_child: Box::new(build_tree(out_edges[0].target, edges, predictions, decisions)?),
                    gt_child: Box::new(build_tree(out_edges[1].target, edges, predictions, decisions)?),
                })
            }
            ah::bail!("Could not find node with id {node_id}");
        }

        let root = build_tree(0, &edges, &predictions, &decisions)?;
        dbg!(&root);

        let mut code = String::new();
        root.write_wgsl(&mut code, 0).context("Writing shader code")?;
        eprintln!("Shader code:\n{code}");

        let out = Ok(Self{root});
        out
    }

    pub fn write_wgsl(&self, out: &mut impl std::fmt::Write) -> Result<(), std::fmt::Error> {
        self.root.write_wgsl(out, 0)
    }

    pub fn from_dir(dir_name: &str) -> ah::Result<Vec<Self>>{
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir_name).context(format!("Opening dir {dir_name}"))? {
            let entry = entry.context("reading dir entry")?;
            if !entry.file_type()?.is_file(){
                continue
            }
            let path = entry.path();
            let file_name = path.file_name().ok_or(ah::anyhow!("File had no file_name"))?;
            if !file_name.to_string_lossy().starts_with("tree_"){
                continue
            }
            let raw_tree = std::fs::read_to_string(&path).context("reading file")?;
            let tree = Self::parse(&raw_tree).context("parsing tree")?;
            out.push(tree)
        }
        Ok(out)
    }
}


#[test]
fn test_decision_tree_parsing(){
    let _dt = DecisionTree::parse(r#"
        digraph Tree {
            node [shape=box, fontname="helvetica"] ;
            edge [fontname="helvetica"] ;
            0 [label="node #0\nx[16] <= 36.255\nsamples = 88\nvalue = [68, 71]\nclass = 1"] ;
            1 [label="node #1\nx[17] <= 0.954\nsamples = 51\nvalue = [68.0, 14.0]\nclass = 0"] ;
            0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
            2 [label="node #2\nsamples = 45\nvalue = [68, 0]\nclass = 0"] ;
            1 -> 2 ;
            3 [label="node #3\nsamples = 6\nvalue = [0, 14]\nclass = 1"] ;
            1 -> 3 ;
            4 [label="node #4\nsamples = 37\nvalue = [0, 57]\nclass = 1"] ;
            0 -> 4 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
        }
    "#).unwrap();
}
