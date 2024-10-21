```bash
wget https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/mesh.nt

cat mesh.nt |  grep -e "^<http://id.nlm.nih.gov/mesh/M" mesh.nt | grep -e "vocab#scopeNote" -e "#label" > mesh_concept.nt
cat mesh.nt |  grep -e "^<http://id.nlm.nih.gov/mesh/D" mesh.nt | grep -e "vocab#annotation" -e "#label" > mesh_descriptor.nt
```


