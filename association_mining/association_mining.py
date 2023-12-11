import math

"""
A program which filters a set of associations based on significance, using a variety of metrics: lift, leverage, mutual information (MI),
value-based interpretation, and conditional MI. 
"""

# The first column shows X for a rule X -> C=c, where c = having Alzheimer's disease;
# The second column displays number of people for whom X is true, while the third column shows number of people for whom both X and C=c are true.
alzheimer_rules = [
    ["smoking", 300, 125],
    ["stress", 500, 150],
    ["higheducation", 500, 100],
    ["tea", 342, 102],
    ["turmeric", 2, 0],
    ["female", 500, 148],
    ["female;stress", 260, 100],
    ["berries;apples", 120, 32],
    ["smoking;tea", 240, 100],
    ["smoking;higheducation", 80, 32],
    ["stress;smoking", 200, 100],
    ["female;higheducation", 251, 48]
]

pruned_rules = []
p_c = 3 / 10  # percentage of people with Alzheimer's disease
p_nc = 1 - p_c  # percentage of people who don't have Alzheimer's disease
n = 1000

# Calculate lift and leverage for each rule and prune out rules with lift < 1 or leverage < 0.
for rule in alzheimer_rules:
    p_x = rule[1] / n  # probability of X
    p_c_x = rule[2] / rule[1]  # probability of Alzheimer's given X
    p_xc = p_x * p_c_x  # probability of both X and having Alzheimer's
    lift = p_xc / p_x / p_c
    leverage = p_xc - p_x * p_c

    if lift >= 1 or leverage >= 0:
        pruned_rules.append(rule)

pruned_rules2 = []

# Calculate mutual information (MI) scores for each rule and remove those for which n * MI < 1.5
for rule in pruned_rules:
    p_x = rule[1] / n
    p_nx = 1 - p_x

    # conditionals
    p_c_x = rule[2] / rule[1]
    p_nc_x = (rule[1] - rule[2]) / rule[1]

    p_xc = p_x * p_c_x
    p_xnc = p_x * p_nc_x

    # conditionals with nx
    p_c_nx = (p_c - p_xc) / p_nx
    p_nc_nx = (p_nc - p_xnc) / p_nx

    p_nxc = p_nx * p_c_nx
    p_nxnc = p_nx * p_nc_nx

    nmi = n * (p_xc * math.log2(p_xc) + p_xnc * math.log2(p_xnc) + p_nxc * math.log2(p_nxc) + p_nxnc * math.log2(p_nxnc) -
               p_x * math.log2(p_x) - p_nx * math.log2(p_nx) - p_c * math.log2(p_c) - p_nc * math.log2(p_nc))

    if nmi >= 1.5:
        pruned_rules2.append(rule)

pruned_rules = pruned_rules2
pruned_rules2 = []

# Remove no-improvement rules (rules of the form XY -> C=c for which P(C|X) >= P(C|XY) or P(C|Y) >= P(C|XY)).
for rule in pruned_rules:
    rule_x = []
    rule_q = []

    if len(rule[0].split(";")) == 1:
        pruned_rules2.append(rule)

        continue

    for rule2 in alzheimer_rules:
        if rule2[0] == rule[0].split(";")[0]:
            rule_x = rule2

        elif rule2[0] == rule[0].split(";")[1]:
            rule_q = rule2

    if not (rule_x[2] / rule_x[1] >= rule[2] / rule[1] or rule_q[2] / rule_q[1] >= rule[2] / rule[1]):
        pruned_rules2.append(rule)

pruned_rules = pruned_rules2
pruned_rules2 = []

# Calculate conditional MI scores for each rule and remove those for which n * MIC < 0.5
for rule in pruned_rules:
    rule_x = []
    rule_q = []

    if len(rule[0].split(";")) == 1:
        pruned_rules2.append(rule)

        continue

    for rule2 in alzheimer_rules:
        if rule2[0] == rule[0].split(";")[0]:
            rule_x = rule2

        elif rule2[0] == rule[0].split(";")[1]:
            rule_q = rule2

    p_xq = rule[1] / n
    p_x = rule_x[1] / n
    p_q = rule_q[1] / n
    p_nq = 1 - p_q

    # conditionals necessary for calculating formula terms
    p_c_xq = rule[2] / rule[1]
    p_nc_xq = (rule[1] - rule[2]) / rule[1]
    p_c_q = rule_q[2] / rule_q[1]
    p_nc_q = (rule_q[1] - rule_q[2]) / rule_q[1]
    p_c_x = rule_x[2] / rule_x[1]
    p_nc_x = (rule_x[1] - rule_x[2]) / rule_x[1]

    p_xnq = p_x - p_xq
    p_qc = p_q * p_c_q
    p_qnc = p_q * p_nc_q
    p_xc = p_x * p_c_x
    p_xnc = p_x * p_nc_x

    # conditionals necessary for calculating formula terms
    p_c_nq = (p_c - p_qc) / p_nq
    p_nc_nq = (p_nc - p_qnc) / p_nq
    p_c_xnq = (rule_x[2] - rule[2]) / (rule_x[1] - rule[1])
    p_nc_xnq = (rule_x[1] - rule[1] - rule_x[2] + rule[2]) / (rule_x[1] - rule[1])

    p_xqc = p_xq * p_c_xq
    p_xqnc = p_xq * p_nc_xq
    p_xnqc = p_xnq * p_c_xnq
    p_xnqnc = p_xnq * p_nc_xnq

    nmic = n * (p_x * math.log2(p_x) + p_xqc * math.log2(p_xqc) + p_xqnc * math.log2(p_xqnc) + p_xnqc * math.log2(p_xnqc) +
                p_xnqnc * math.log2(p_xnqnc) - p_xq * math.log2(p_xq) - p_xnq * math.log2(p_xnq) - p_xc * math.log2(p_xc) -
                p_xnc * math.log2(p_xnc))

    if nmic >= 0.5:
        pruned_rules2.append(rule)

print(pruned_rules2)
