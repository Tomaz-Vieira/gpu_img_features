use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{self as ah, Context};

use graphviz_rust as gv;
use graphviz_rust::dot_structures as gs;

pub struct DecisionTree{
    stmts: Vec<gs::Stmt>,
}

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

#[derive(Debug)]
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

enum DecisionNode{
    Decision{
        decision: Decision,
        le_child: Box<DecisionNode>,
        gt_child: Box<DecisionNode>,
    },
    Prediction(Prediction),
}

fn parse_vert(vert: &gs::Vertex) -> ah::Result<u32>{
    let gs::Vertex::N(node) = vert else{
        return Err(ah::anyhow!("Vert is not a node: {vert:?}"))
    }; 
    let node_id: &gs::Id = &node.0;
    node_id.to_string().parse::<u32>().context("Parsing vertex id")
}

fn parse_edge(edge: &gs::Edge) -> ah::Result<(u32, u32)>{
    let gs::EdgeTy::Pair(v1, v2) = &edge.ty else {
        return Err(ah::anyhow!("Don't know how to handle non-pair edges"))
    };
    Ok((
        parse_vert(v1)?,
        parse_vert(v2)?
    ))
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
        for s in &stmts{
            let gs::Stmt::Node(node) = s else {
                continue;
            };
            let node_id = node.id.0.to_string().parse::<u32>().context("Parsing node Id of {node:?}")?;
            let is_leaf = edges.iter().find(|(orig, _tgt)| *orig == node_id).is_some();
            let Some(label_attr) = node.attributes.iter().find(|attr| attr.0.to_string() == "label") else {
                ah::bail!("Node has no label {node:?}");
            };
            let raw_label = label_attr.1.to_string();
            let trimmed_label = raw_label.trim_matches('"');
            for label_attr in trimmed_label.split("\\n"){
                if is_leaf{
                    predictions.insert(node_id, Prediction::try_parse_label_attr(&label_attr)?{
                        predictions.insert(node_id, prediction);
                    }
                }
                if Decision::try_parse_label_attr(&label_attr)?.is_some(){ //decision nodes could be parsed as predicitons
                    continue
                }
            }
        }

        dbg!(predictions);

        Ok(Self{stmts})
    }

    pub fn write_wgsl(&self, out: &mut impl std::fmt::Write){
        for stmt in &self.stmts{
            // dbg!(stmt);
        }
    }
}


#[test]
fn test_decision_tree_parsing(){
    let dt = DecisionTree::parse(r#"
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

    let mut out = String::new();
    dt.write_wgsl(&mut out);
}
