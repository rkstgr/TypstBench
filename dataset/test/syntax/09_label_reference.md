---
id: reference-01
tier: basic
task_type: conversion
features: ["labels", "references"]
difficulty: 2
---
Set the heading to be numbered like "1.", create a level 1 heading called "Methods" with a label "methods", then create text "As described in", followed by the refernce to the label, 
---
```typst
#set heading(numbering: "1.")
= Methods <methods>
As described in @methods
```