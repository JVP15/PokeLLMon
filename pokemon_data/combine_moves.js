// using this file to extract pokemon data Pokemon Showdown and into a format I can use a bit more readily
// I expect that pokemon showdown will be downloaded in the parent directory

const fs = require('fs');

console.log(__dirname);

// Import Moves and MovesText data
const moves = require(__dirname + '\\..\\pokemon-showdown\\dist\\data\\moves.js').Moves;
const movesText = require(__dirname + '\\..\\pokemon-showdown\\dist\\data\\text\\moves.js').MovesText;

// Combine the two dictionaries
for (let move in moves) {
    if (movesText[move]) {
        moves[move] = {...moves[move], ...movesText[move]};
    }
}

// Write the combined dictionary to a new JSON file
fs.writeFileSync('./moves_text.json', JSON.stringify(moves, null, 2));


const abilitiesText = require(__dirname + '\\..\\pokemon-showdown\\dist\\data\\text\\abilities.js').AbilitiesText;

fs.writeFileSync('./abilities_text.json', JSON.stringify(abilitiesText, null, 2));

const itemsText = require(__dirname + '\\..\\pokemon-showdown\\dist\\data\\text\\items.js').ItemsText;

fs.writeFileSync('./items_text.json', JSON.stringify(itemsText, null, 2));


const pokedex = require(__dirname + '\\..\\pokemon-showdown\\dist\\data\\pokedex.js').Pokedex;

fs.writeFileSync('./pokedex.json', JSON.stringify(pokedex, null, 2));


const types = require(__dirname + '\\..\\pokemon-showdown\\dist\\data\\typechart.js').TypeChart;

fs.writeFileSync('./typechart.json', JSON.stringify(types, null, 2));





