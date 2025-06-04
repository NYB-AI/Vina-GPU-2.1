/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/

#ifndef VINA_BOX_UTILS_H
#define VINA_BOX_UTILS_H

#include "model.h"
#include "grid_dim.h"
#include "common.h"
#include "brick.h"

// Determines if an atom is within a certain distance from the search box
inline bool atom_within_box_distance(const atom& a, const grid_dims& gd, fl margin) {
    vec box_begin(gd[0].begin, gd[1].begin, gd[2].begin);
    vec box_end(gd[0].end, gd[1].end, gd[2].end);
    
    // Use brick_distance_sqr to calculate the squared distance from the atom to the box
    return brick_distance_sqr(box_begin, box_end, a.coords) <= (margin * margin);
}

// Filter a model to keep only atoms within the specified margin from the search box
struct filtered_model {
    std::vector<atom> grid_atoms;
    std::vector<sz> original_indices;
    
    filtered_model(const model& m, const grid_dims& gd, fl margin) {
        const sz nat = num_atom_types(m.atom_typing_used());
        
        // Filter grid atoms based on distance to the search box
        VINA_FOR_IN(i, m.grid_atoms) {
            const atom& a = m.grid_atoms[i];
            if (a.get(m.atom_typing_used()) < nat && atom_within_box_distance(a, gd, margin)) {
                grid_atoms.push_back(a);
                original_indices.push_back(i);
            }
        }
    }
    
    // Get the number of atoms removed through filtering
    sz num_removed_atoms(const model& original) const {
        return original.grid_atoms.size() - grid_atoms.size();
    }
    
    // Get the percentage of atoms removed
    fl percentage_removed(const model& original) const {
        return 100.0 * (1.0 - (fl)grid_atoms.size() / (fl)original.grid_atoms.size());
    }
};

#endif // VINA_BOX_UTILS_H 