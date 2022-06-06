#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define NB_SIZE 8

typedef struct {
    bool rule[NB_SIZE + 1];
}rules;

rules* getAllRules() {
    int number_of_rules = 1;
    for (int i = 0; i < NB_SIZE + 1; i++) {
        number_of_rules *= 2;
    }

    rules* allRules = (rules*)malloc(sizeof(rules) * number_of_rules);

    int counter = 0;

    FILE* ruleFile;
    const char* rulePath = "data/rules/rules.csv";

    ruleFile = fopen(rulePath, "r");

    for (int i = 0; i < NB_SIZE + 1; i++) {
        allRules[counter].rule[i] = false;
    }
    while (1) {

        char c = fgetc(ruleFile);
        if (c == EOF) {
            break;
        }
        else if (c == '\n') {
            counter++;
            for (int i = 0; i < NB_SIZE + 1; i++) {
                allRules[counter].rule[i] = false;
            }
        }
        else if (c != ',' && c != ' ') {
            allRules[counter].rule[(int)c - 48] = true;
        }
    }
    return allRules;
}