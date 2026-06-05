//
//  io.cpp
//  adaptive_colume_grid
//
//  Created by Yiwen Ju on 12/3/24.
//
#include "io.h"
#include <sweep/logger.h>
#include "col_gridgen.h"
#include <queue>


void convert_4d_grid_col(
    mtet::MTetMesh grid,
    vertExtrude vertexMap,
    std::vector<std::array<double, 3>>& verts,
    std::vector<std::array<size_t, 4>>& simps,
    std::vector<std::vector<double>>& time,
    std::vector<std::vector<double>>& values)
{
    size_t vert_num = grid.get_num_vertices();
    size_t tet_num = grid.get_num_tets();
    verts.reserve(vert_num);
    simps.reserve(tet_num);
    time.reserve(vert_num);
    values.reserve(vert_num);
    size_t vertIt = 0;
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap ind4DMap;
    grid.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        verts.emplace_back(std::array<double, 3>{data[0], data[1], data[2]});
        vertexCol::vert4d_list vert4dList = vertexMap[value_of(vid)].vert4dList;
        ind4DMap[value_of(vid)] = vertIt;
        values.emplace_back(std::vector<double>{});
        time.emplace_back(std::vector<double>{});
        values[vertIt].reserve(vert4dList.size());
        time[vertIt].reserve(vert4dList.size());
        for (size_t i = 0; i < vert4dList.size(); i++) {
            Eigen::RowVector4d coord = vert4dList[i].coord;
            values[vertIt].emplace_back(vert4dList[i].valGradList.second[3]);
            time[vertIt].emplace_back(vert4dList[i].coord(3));
        }
        vertIt++;
    });
    grid.seq_foreach_tet([&](TetId tid, [[maybe_unused]] std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = grid.get_tet(tid);
        simps.emplace_back(std::array<size_t, 4>{
            ind4DMap[value_of(vs[0])],
            ind4DMap[value_of(vs[1])],
            ind4DMap[value_of(vs[2])],
            ind4DMap[value_of(vs[3])]});
    });
}

uint64_t getTetKeyByVidsIO(const std::span<VertexId, 4>& vs)
{
    std::array<uint64_t, 4> ids = {value_of(vs[0]), value_of(vs[1]), value_of(vs[2]), value_of(vs[3])};
    std::sort(ids.begin(), ids.end());
    // combine into single hash
    uint64_t key = ids[0];
    key ^= ids[1] + 0x9e3779b9 + (key << 6) + (key >> 2);
    key ^= ids[2] + 0x9e3779b9 + (key << 6) + (key >> 2);
    key ^= ids[3] + 0x9e3779b9 + (key << 6) + (key >> 2);
    return key;
}

void convert_4d_grid_mtetcol(
    mtet::MTetMesh grid,
    vertExtrude vertexMap,
    std::unordered_map<uint64_t, int>& activeColMap,
    std::unordered_map<uint64_t, int>& markTetMap,
    std::unordered_map<uint64_t, std::vector<size_t>>& markTetActive4DtetIdsMap,
    std::vector<double>& verts,
    std::vector<size_t>& simps,
    std::vector<int>& tetActiveTags,
    std::vector<int>& tetMarkTags,
    std::vector<std::vector<size_t>>& tetMarkActive4DtetIds,
    std::vector<std::vector<double>>& time,
    const std::string& out_dir,
    bool cyclic)
{
    size_t vert_num = grid.get_num_vertices();
    size_t tet_num = grid.get_num_tets();
    size_t tet4d_num = 0, vert4d_num = 0;
    verts.reserve(vert_num * 3);
    simps.reserve(tet_num * 4);
    tetActiveTags.reserve(tet_num);
    time.reserve(vert_num);
    size_t vertIt = 0;
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap ind4DMap;
    grid.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        verts.emplace_back(static_cast<double>(data[0]));
        verts.emplace_back(static_cast<double>(data[1]));
        verts.emplace_back(static_cast<double>(data[2]));
        vertexCol::vert4d_list vert4dList = vertexMap[value_of(vid)].vert4dList;
        ind4DMap[value_of(vid)] = vertIt;
        time.emplace_back(std::vector<double>{});
        time[vertIt].reserve(vert4dList.size());
        vert4d_num += vert4dList.size();
        for (size_t i = 0; i < vert4dList.size(); i++) {
            Eigen::RowVector4d coord = vert4dList[i].coord;
            // values[vertIt].emplace_back(vert4dList[i].valGradList.second[3]);
            time[vertIt].emplace_back(vert4dList[i].coord(3));
        }
        vertIt++;
    });
    size_t active_tet_count = 0;
    size_t mark_tet_count = 0;
    bool mark_one_ring_tets = true;
    int mark_ring_depth = 0;
    std::unordered_set<uint64_t> marked_tet_keys;

    auto get_tet_key = [&](TetId tid) {
        std::span<VertexId, 4> tvs = grid.get_tet(tid);
        return getTetKeyByVidsIO(tvs);
    };

    auto is_active_tet_key = [&](uint64_t tetKey) {
        return activeColMap.find(tetKey) != activeColMap.end();
    };

    auto collect_neighboring_tets_by_edge = [&](mtet::EdgeId eid) {
        std::vector<TetId> neighbor_tets;

        grid.foreach_tet_around_edge(eid, [&](TetId t) {
            uint64_t tetKey = get_tet_key(t);

            // Preserve your current behavior: only expand into active tets.
            if (is_active_tet_key(tetKey)) {
                neighbor_tets.push_back(t);
            }
        });

        return neighbor_tets;
    };

    auto collect_neighboring_tets_by_tet = [&](TetId tid) {
        std::vector<TetId> neighbor_tets;

        grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId, mtet::VertexId) {
            auto edge_neighbors = collect_neighboring_tets_by_edge(eid);
            neighbor_tets.insert(
                neighbor_tets.end(),
                edge_neighbors.begin(),
                edge_neighbors.end());
        });

        return neighbor_tets;
    };

    std::vector<TetId> frontier;

    grid.seq_foreach_tet([&](TetId tid, [[maybe_unused]] std::span<const VertexId, 4> data) {
        uint64_t tetKey = get_tet_key(tid);

        if (markTetMap.find(tetKey) != markTetMap.end()) {
            marked_tet_keys.insert(tetKey);
            frontier.push_back(tid);
        }
    });

    for (int ring = 0; ring < mark_ring_depth; ++ring) {
        std::vector<TetId> next_frontier;

        for (TetId tid : frontier) {
            auto neighbor_tets = collect_neighboring_tets_by_tet(tid);

            for (TetId neighbor_tid : neighbor_tets) {
                uint64_t neighbor_key = get_tet_key(neighbor_tid);

                auto [_, inserted] = marked_tet_keys.insert(neighbor_key);
                if (inserted) {
                    next_frontier.push_back(neighbor_tid);
                }
            }
        }

        frontier = std::move(next_frontier);

        if (frontier.empty()) {
            break;
        }
    }
    // std::unordered_set<uint64_t> marked_tet_keys;
    // auto collect_neighboring_tets_by_edge = [&](mtet::EdgeId eid) 
    //     {
    //         std::vector<uint64_t> neig_tet_keys;
    //         grid.foreach_tet_around_edge(eid, [&](mtet::TetId t) {
    //             std::span<VertexId, 4> tvs = grid.get_tet(t);
    //             auto tetKey = getTetKeyByVidsIO(tvs);
    //             if(activeColMap.find(tetKey) != activeColMap.end())
    //             neig_tet_keys.push_back(tetKey); 
    //         }); 
    //         return neig_tet_keys;
    //     };

    // auto collect_neighboring_tets_by_tet = [&](TetId tid){
    //     std::vector<uint64_t> neig_tet_keys;
    //     grid.foreach_edge_in_tet(tid, [&](mtet::EdgeId eid, mtet::VertexId v0, mtet::VertexId v1) {
    //         auto cur_neig_tets = collect_neighboring_tets_by_edge(eid); 
    //         for(auto cur_key : cur_neig_tets)
    //         {
    //             // if(activeColMap.find(cur_key) != activeColMap.end())
    //             neig_tet_keys.push_back(cur_key);
    //         }                             
    //     });
    //     return neig_tet_keys;
    // };
    // grid.seq_foreach_tet([&](TetId tid, [[maybe_unused]] std::span<const VertexId, 4> data) {
    //     std::span<VertexId, 4> vs = grid.get_tet(tid);
    //     auto tetKey = getTetKeyByVidsIO(vs);
    //     auto it = markTetMap.find(tetKey);
    //     if(it != markTetMap.end())
    //     {
    //         marked_tet_keys.insert(tetKey);
    //         if(mark_one_ring_tets)
    //         {
    //             auto neighbor_tet_keys = collect_neighboring_tets_by_tet(tid);
    //             for(auto key : neighbor_tet_keys)
    //             {
    //                 marked_tet_keys.insert(key);
    //             }
    //         } 
    //     }
    // });
    grid.seq_foreach_tet([&](TetId tid, [[maybe_unused]] std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = grid.get_tet(tid);
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[0])]));
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[1])]));
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[2])]));
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[3])]));
       
        auto tetKey = getTetKeyByVidsIO(vs);
        auto it = activeColMap.find(tetKey);
        if(it != activeColMap.end()) active_tet_count ++;
        int tetActiveVal = (it != activeColMap.end()) ? it->second : 0;
        tetActiveTags.push_back(tetActiveVal); 

        auto it2 = marked_tet_keys.find(tetKey);
        int tetMarkVal =  0;
        if(it2 != marked_tet_keys.end())
        {
            mark_tet_count ++;
            tetMarkVal = 1;
        }
        tetMarkTags.push_back(tetMarkVal);
        std::vector<size_t> active4DtetIds = {};
        if(markTetActive4DtetIdsMap.find(tetKey) != markTetActive4DtetIdsMap.end())
        {
            active4DtetIds = markTetActive4DtetIdsMap[tetKey];
        }
        tetMarkActive4DtetIds.push_back(active4DtetIds);
        
        tet4d_num += vertexMap[value_of(vs[0])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[1])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[2])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[3])].vert4dList.size();
        tet4d_num -= 4;
    });

    
    std::string log_path = out_dir + "/run_log.txt";
    std::ofstream log_file(log_path, std::ios::app);
   
    if (log_file.is_open()) 
    {
        log_file << "3D grid tet Number:  "+ std::to_string(grid.get_num_tets()) 
                +  " 3D active Tet Number: "+ std::to_string(active_tet_count) << std::endl;
        log_file << "4D Vertex Number:  "+ std::to_string(vert4d_num) 
                +  " 4D Tetrahedra Number: "+ std::to_string(tet4d_num) << std::endl;
        log_file.close();
    }
    
    

    sweep::logger().info("3D grid tet Number: {} 3D active Tet Number: {}", grid.get_num_tets(), active_tet_count);

    sweep::logger().info("4D Vertex Number: {} 4D Tetrahedra Number: {}", vert4d_num, tet4d_num);
}


void convert_4d_grid_mtetcol(
    mtet::MTetMesh& grid,
    vertExtrude& vertexMap,
    std::vector<double>& verts4d,       // flat: x,y,z,t per vertex
    std::vector<size_t>& tets4d        // flat: 5 indices per 4D tet (pentatope))
    )
{
    using namespace mtet;

    // --- Phase 1: Build global 4D vertex list ---
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap base3dMap;

    std::vector<size_t> vertStart;
    vertStart.reserve(grid.get_num_vertices());

    size_t vert4d_num = 0;

    grid.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        base3dMap[value_of(vid)] = base3dMap.size();

        const auto& vert4dList = vertexMap[value_of(vid)].vert4dList;
        vertStart.push_back(vert4d_num);

        for (size_t i = 0; i < vert4dList.size(); i++) {
            const Eigen::RowVector4d& coord = vert4dList[i].coord;
            verts4d.push_back(coord(0));
            verts4d.push_back(coord(1));
            verts4d.push_back(coord(2));
            verts4d.push_back(coord(3));
        }

        vert4d_num += vert4dList.size();
    });

    // --- Phase 2: For each 3D tet, sample the column and emit 4D tets ---
    size_t tet4d_num = 0;
    size_t active_tet_count = 0;

    grid.seq_foreach_tet([&](TetId tid, [[maybe_unused]] std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = grid.get_tet(tid);

        auto tetKey = getTetKeyByVidsIO(vs);
        simpCol::cell5_list cell5Col;
        sampleCol(vs, vertexMap, cell5Col);
        const std::array<size_t, 4> voff = {
            vertStart[base3dMap[value_of(vs[0])]],
            vertStart[base3dMap[value_of(vs[1])]],
            vertStart[base3dMap[value_of(vs[2])]],
            vertStart[base3dMap[value_of(vs[3])]]};

        for (const auto& c5 : cell5Col) {
            const int* idx = c5.hash.data();
            int tag = idx[4];
            // 5 vertices of the pentatope (from bind_cell5_verts logic):
            //   [0] = advancing vertex, current time sample
            //   [1..3] = held vertices (the other 3 base verts)
            //   [4] = advancing vertex, previous time sample
            std::array<size_t, 5> gid;
            gid[0] = voff[tag] + idx[tag];
            size_t slot = 1;
            for (int b = 0; b < 4; b++) {
                if (b != tag)
                    gid[slot++] = voff[b] + idx[b];
            }
            gid[4] = voff[tag] + idx[tag] - 1;

            for (int v = 0; v < 5; v++)
                tets4d.push_back(gid[v]);
            tet4d_num++;
        }
    });
    // --- Logging ---
    sweep::logger().info("3D grid tet Number: {} 3D active: {}", grid.get_num_tets(), active_tet_count);
    sweep::logger().info("4D Vertices: {} 4D Tets: {}", vert4d_num, tet4d_num);
}



void convert_4d_grid_mtetcol(
    mtet::MTetMesh grid,
    vertExtrude vertexMap,
    std::vector<double>& verts,
    std::vector<size_t>& simps,
    std::vector<std::vector<double>>& time,
    std::vector<std::vector<double>>& values,
    bool cyclic)
{
    size_t vert_num = grid.get_num_vertices();
    size_t tet_num = grid.get_num_tets();
    size_t tet4d_num = 0, vert4d_num = 0;
    verts.reserve(vert_num * 3);
    simps.reserve(tet_num * 4);
    time.reserve(vert_num);
    values.reserve(vert_num);
    size_t vertIt = 0;
    using IndexMap = ankerl::unordered_dense::map<uint64_t, size_t>;
    IndexMap ind4DMap;
    grid.seq_foreach_vertex([&](VertexId vid, std::span<const Scalar, 3> data) {
        verts.emplace_back(static_cast<double>(data[0]));
        verts.emplace_back(static_cast<double>(data[1]));
        verts.emplace_back(static_cast<double>(data[2]));
        vertexCol::vert4d_list vert4dList = vertexMap[value_of(vid)].vert4dList;
        ind4DMap[value_of(vid)] = vertIt;
        values.emplace_back(std::vector<double>{});
        time.emplace_back(std::vector<double>{});
        values[vertIt].reserve(vert4dList.size());
        time[vertIt].reserve(vert4dList.size());
        vert4d_num += vert4dList.size();
        for (size_t i = 0; i < vert4dList.size(); i++) {
            Eigen::RowVector4d coord = vert4dList[i].coord;
            values[vertIt].emplace_back(vert4dList[i].valGradList.second[3]);
            time[vertIt].emplace_back(vert4dList[i].coord(3));
        }
        if (cyclic) {
            values[vertIt].back() = values[vertIt].front();
        }
        vertIt++;
    });
    grid.seq_foreach_tet([&](TetId tid, [[maybe_unused]] std::span<const VertexId, 4> data) {
        std::span<VertexId, 4> vs = grid.get_tet(tid);
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[0])]));
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[1])]));
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[2])]));
        simps.emplace_back(static_cast<size_t>(ind4DMap[value_of(vs[3])]));
        tet4d_num += vertexMap[value_of(vs[0])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[1])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[2])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[3])].vert4dList.size();
        tet4d_num -= 4;
    });
    sweep::logger().info("4D Vertex Number: {} 4D Tetrahedra Number: {}", vert4d_num, tet4d_num);
}




mshio::MshSpec generate_spec(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, TimeMap timeMap)
{
    size_t num_vertices = V.rows();
    size_t num_cycles = F.rows();
    mshio::MshSpec spec;
    spec.mesh_format.file_type = 1; // binary
    // Initialize nodes
    auto& nodes = spec.nodes;
    nodes.num_entity_blocks = 1;
    nodes.num_nodes = num_vertices;
    nodes.min_node_tag = 1;
    nodes.max_node_tag = nodes.num_nodes;
    nodes.entity_blocks.resize(1);
    auto& node_block = nodes.entity_blocks[0];
    node_block.entity_dim = 2;
    node_block.entity_tag = 1;
    node_block.parametric = 0;
    node_block.num_nodes_in_block = nodes.num_nodes;
    node_block.tags.reserve(nodes.num_nodes);
    node_block.data.reserve(nodes.num_nodes * 3);
    for (size_t i = 0; i < num_vertices; i++) {
        auto pos = V.row(i);
        node_block.tags.push_back(i + 1);
        node_block.data.push_back(pos[0]);
        node_block.data.push_back(pos[1]);
        node_block.data.push_back(pos[2]);
    }
    mshio::Data node_data;
    node_data.header.string_tags.push_back("time");
    node_data.header.int_tags.push_back(0);
    node_data.header.int_tags.push_back(1);
    node_data.header.int_tags.push_back(num_vertices);
    auto& entries = node_data.entries;
    double tempTime = 0;
    entries.resize(num_vertices);
    for (size_t i = 0; i < num_vertices; i++) {
        Eigen::RowVector3d pos = V.row(i);
        auto key = QuantizedRowVec3::from(pos);
        auto it = find(timeMap, pos);
        if (it != timeMap.end()) {
            double time = timeMap[key];
            tempTime = time;
        } else {
            std::cout << "WARNING: NOT FOUND" << std::endl;
        }
        auto& entry = entries[i];
        entry.tag = i + 1;
        entry.data = {tempTime};
    }
    spec.node_data.push_back(std::move(node_data));
    // Initialize elements
    auto& elements = spec.elements;
    elements.num_entity_blocks = 1;
    elements.num_elements = num_cycles;
    elements.min_element_tag = 1;
    elements.max_element_tag = elements.num_elements;
    elements.entity_blocks.resize(1);
    auto& element_block = elements.entity_blocks[0];
    element_block.entity_dim = 2;
    element_block.entity_tag = 1;
    element_block.element_type = 2;
    element_block.num_elements_in_block = elements.num_elements;
    element_block.data.reserve(elements.num_elements * 4);
    for (size_t i = 0; i < num_cycles; i++) {
        auto cycle = F.row(i);
        assert(cycle.size() == 3);
        element_block.data.push_back(i + 1);
        for (auto si : cycle) {
            element_block.data.push_back(si + 1);
        }
    }
    return spec;
}

// Build vertex adjacency (undirected) from triangular faces
static std::vector<std::vector<int>> build_adjacency(const Eigen::MatrixXi& F, int nV)
{
    std::vector<std::vector<int>> adj(nV);
    auto add_undirected = [&](int a, int b) {
        if (a == b) return;
        adj[a].push_back(b);
        adj[b].push_back(a);
    };
    for (int f = 0; f < F.rows(); ++f) {
        int a = F(f, 0), b = F(f, 1), c = F(f, 2);
        add_undirected(a, b);
        add_undirected(b, c);
        add_undirected(c, a);
    }
    return adj;
}

/**
 * Multi-source BFS label propagation (no averaging).
 * - Start from all labeled vertices (from timeMap) as seeds.
 * - For each unlabeled neighbor, assign the seed's label and push to queue.
 * - Continue BFS until all reachable vertices are labeled.
 * - If multiple seeds could reach a vertex, the one dequeued first wins.
 *   (Deterministic if we push seeds in ascending index order.)
 *
 * @returns per-vertex labels (size n). Unreachable or no-seed case => NaN.
 */
Eigen::VectorXd propagate_labels_bfs(
    const Eigen::MatrixXd& V, // n x 3
    const Eigen::MatrixXi& F, // m x 3
    const TimeMap& timeMap)
{
    const int n = static_cast<int>(V.rows());
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    const double INF = std::numeric_limits<double>::infinity();

    // 1) Initialize labels from timeMap
    Eigen::VectorXd L(n);
    L.setConstant(NaN);
    for (int i = 0; i < n; ++i) {
        auto it = timeMap.find(QuantizedRowVec3::from(V.row(i)));
        if (it != timeMap.end()) L[i] = it->second;
    }

    // 2) Adjacency
    auto adj = build_adjacency(F, n);

    // 3) Visit unlabeled components
    std::vector<char> visited(n, 0);
    std::vector<int> comp;
    comp.reserve(256);

    for (int s = 0; s < n; ++s) {
        if (!std::isnan(L[s]) || visited[s]) continue; // skip labeled or already handled

        // BFS over unlabeled-only region starting at s
        comp.clear();
        std::queue<int> q;
        q.push(s);
        visited[s] = 1;

        double minBoundary = INF;

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            comp.push_back(v);

            for (int nb : adj[v]) {
                if (std::isnan(L[nb])) {
                    if (!visited[nb]) {
                        visited[nb] = 1;
                        q.push(nb);
                    }
                } else {
                    // neighbor is labeled => boundary candidate
                    minBoundary = std::min(minBoundary, L[nb]);
                }
            }
        }

        // Assign the component
        if (minBoundary != INF) {
            for (int v : comp) L[v] = minBoundary;
        } // else: no labeled neighbor — leave NaN
    }

    return L;
}


// Optional: write propagated labels back into timeMap (insert only if missing)
void backfill_timeMap_from_labels(
    const Eigen::MatrixXd& V,
    const Eigen::VectorXd& L,
    TimeMap& timeMap)
{
    for (int i = 0; i < V.rows(); ++i) {
        if (std::isnan(L[i]))
            continue; // no seed reachable (shouldn't happen if connected & has seeds)
        auto key = QuantizedRowVec3::from(V.row(i));
        if (timeMap.find(key) == timeMap.end()) {
            //      timeMap.emplace(key, L[i]);
            timeMap[key] = L[i];
            //        std::cout << "Propagated" << std::endl;
        }
    }
}

void save_grid_for_mathematica(
    std::string_view filename,
    mtet::MTetMesh grid,
    vertExtrude vertexMap)
{
    /// Mathematica isosurfacing output:
    std::vector<std::array<double, 3>> verts_math;
    std::vector<std::array<size_t, 4>> simps_math;
    std::vector<std::vector<double>> time_math;
    std::vector<std::vector<double>> values_math;
    convert_4d_grid_col(grid, vertexMap, verts_math, simps_math, time_math, values_math);
    {
        using json = nlohmann::json;
        std::ofstream fout(filename.data(), std::ios::out);
        if (!fout) {
            throw std::runtime_error("Failed to open file for writing");
        }
        json jOut;
        jOut.push_back(json(verts_math));
        jOut.push_back(json(simps_math));
        jOut.push_back(json(time_math));
        jOut.push_back(json(values_math));
        fout << jOut.dump(4, ' ', true, json::error_handler_t::replace) << std::endl;
    }
    /// End of Mathematica output
}

void export_to_mathematica(
    const std::string& filename,
    const std::vector<double>& verts,
    const std::vector<size_t>& simps,
    const std::vector<int>& activeTags,
    const std::vector<std::vector<double>>& time)
{
    std::ofstream f(filename);
    f << std::setprecision(16);

    // verts: flat list of xyz, output as {{x,y,z},{x,y,z},...}
    f << "verts = {";
    for (size_t i = 0; i < verts.size(); i += 3) {
        f << "{" << verts[i] << "," << verts[i+1] << "," << verts[i+2] << "}";
        if (i + 3 < verts.size()) f << ",";
    }
    f << "};\n";

    // simps: flat list of indices, output as {{a,b,c,d},...}
    f << "simps = {";
    for (size_t i = 0; i < simps.size(); i += 4) {
        // +1 for Mathematica 1-based indexing
        f << "{" << simps[i]+1 << "," << simps[i+1]+1 << "," << simps[i+2]+1 << "," << simps[i+3]+1 << "}";
        if (i + 4 < simps.size()) f << ",";
    }
    f << "};\n";

    // time: list of lists
    f << "time = {";
    for (size_t i = 0; i < time.size(); i++) {
        f << "{";
        for (size_t j = 0; j < time[i].size(); j++) {
            f << time[i][j];
            if (j + 1 < time[i].size()) f << ",";
        }
        f << "}";
        if (i + 1 < time.size()) f << ",";
    }
    f << "};\n";

    // simps: flat list of indices, output as {{a,b,c,d},...}
    f << "activeTags = {";
    for (size_t i = 0; i < activeTags.size(); ++i ) {
        // +1 for Mathematica 1-based indexing
        if( i == activeTags.size() - 1)
        {
            f << activeTags[i];
        } else {
            f << activeTags[i] << ",";
        }
    }
    f << "};\n";

    // values: list of lists
    // f << "values = {";
    // for (size_t i = 0; i < values.size(); i++) {
    //     f << "{";
    //     for (size_t j = 0; j < values[i].size(); j++) {
    //         f << values[i][j];
    //         if (j + 1 < values[i].size()) f << ",";
    //     }
    //     f << "}";
    //     if (i + 1 < values.size()) f << ",";
    // }
    // f << "};\n";
}


void writeGridToJson(
    const std::string& filename,
    const std::vector<double>& verts,
    const std::vector<size_t>& simps,
    const std::vector<int>& activeTags,
    const std::vector<std::vector<double>>& time) 
{
    using json = nlohmann::json;
    json j;
    j["vertices"]           = verts;
    j["simplicies"]         = simps;
    std::vector<double> timeSamples; 
    std::vector<size_t> timeStartIndices;
    timeStartIndices.push_back(0);
    size_t timeEndIndex = 0;
    for(size_t i = 0; i < time.size(); ++i)
    {
        for(auto tVal : time[i])
        {
            timeSamples.push_back(tVal);
        }
        timeEndIndex += time[i].size();
        timeStartIndices.push_back(timeEndIndex);
    }
    j["time_samples"]       = timeSamples;
    j["time_start_indices"] = timeStartIndices;
    j["tet_flags"] = activeTags;
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    file << j.dump(4);  // pretty print with 4-space indent
    file.close();

    std::cout << "Written to " << filename << std::endl;
}


void write_tets_to_ply(
    const std::vector<double>& verts,
    const std::vector<size_t>& simps,
    const std::string& filepath)
{
    assert(verts.size() % 3 == 0);
    assert(simps.size() % 4 == 0);

    const size_t num_verts = verts.size() / 3;
    const size_t num_tets  = simps.size() / 4;
    const size_t num_faces = num_tets * 4;

    std::ofstream ply(filepath, std::ios::out);
    if (!ply.is_open())
        throw std::runtime_error("Cannot open PLY file for writing: " + filepath);

    // ── Header ──────────────────────────────────────────────────────────────
    ply << "ply\n"
        << "format ascii 1.0\n"
        << "comment tet mesh\n"
        << "element vertex " << num_verts << "\n"
        << "property double x\n"
        << "property double y\n"
        << "property double z\n"
        << "element face " << num_faces << "\n"
        << "property list uchar int vertex_indices\n"
        << "end_header\n";

    // ── Vertices ─────────────────────────────────────────────────────────────
    ply << std::fixed << std::setprecision(10);
    for (size_t i = 0; i < num_verts; ++i)
        ply << verts[3*i] << ' ' << verts[3*i+1] << ' ' << verts[3*i+2] << '\n';

    // ── Faces (4 triangles per tet) ─────────────────────────────────────────
    for (size_t i = 0; i < num_tets; ++i)
    {
        size_t v0 = simps[4*i];
        size_t v1 = simps[4*i+1];
        size_t v2 = simps[4*i+2];
        size_t v3 = simps[4*i+3];

        ply << "3 " << v0 << ' ' << v1 << ' ' << v2 << '\n';
        ply << "3 " << v0 << ' ' << v1 << ' ' << v3 << '\n';
        ply << "3 " << v0 << ' ' << v2 << ' ' << v3 << '\n';
        ply << "3 " << v1 << ' ' << v2 << ' ' << v3 << '\n';
    }
}

void save_4d_mesh_binary(
    const std::vector<double>& verts4d,
    const std::vector<size_t>& tets4d,
    const std::string& filepath)
{
    std::ofstream out(filepath, std::ios::binary);

    const size_t num_verts = verts4d.size() / 4;
    const size_t num_tets  = tets4d.size() / 5;

    // Header: vertex count, tet count
    out.write(reinterpret_cast<const char*>(&num_verts), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&num_tets),  sizeof(size_t));

    // Vertex data: num_verts * 4 doubles
    out.write(reinterpret_cast<const char*>(verts4d.data()),
              verts4d.size() * sizeof(double));

    // Tet data: num_tets * 5 size_t
    out.write(reinterpret_cast<const char*>(tets4d.data()),
              tets4d.size() * sizeof(size_t));

    out.close();
}

void load_4d_mesh_binary(
    std::vector<double>& verts4d,
    std::vector<size_t>& tets4d,
    const std::string& filepath)
{
    std::ifstream in(filepath, std::ios::binary);

    size_t num_verts, num_tets;
    in.read(reinterpret_cast<char*>(&num_verts), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&num_tets),  sizeof(size_t));

    verts4d.resize(num_verts * 4);
    tets4d.resize(num_tets * 5);

    in.read(reinterpret_cast<char*>(verts4d.data()),
            verts4d.size() * sizeof(double));
    in.read(reinterpret_cast<char*>(tets4d.data()),
            tets4d.size() * sizeof(size_t));

    in.close();
}

void save_column_mesh_binary(
    const std::vector<double>&              verts,
    const std::vector<size_t>&              simps,
    const std::vector<std::vector<double>>& time,
    const std::vector<int>&                 tetMarkTags,
    const std::vector<std::vector<size_t>>& tetActive4DtetIds,
    const std::string&                      filepath)
{
    std::ofstream out(filepath, std::ios::binary);

    const size_t num_verts = verts.size() / 3;
    const size_t num_tets  = simps.size() / 4;

    out.write(reinterpret_cast<const char*>(&num_verts), sizeof(size_t));
    // out.write(reinterpret_cast<const char*>(&num_tets),  sizeof(size_t));

    out.write(reinterpret_cast<const char*>(verts.data()),
              verts.size() * sizeof(double));

    // out.write(reinterpret_cast<const char*>(simps.data()),
    //           simps.size() * sizeof(size_t));

    for (size_t i = 0; i < num_verts; ++i) {
        size_t len = time[i].size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(time[i].data()),
                  len * sizeof(double));
    }

    std::vector<size_t> marked_tets;
    std::vector<size_t> unmarked_tets;
    std::vector<std::vector<size_t>> markTetActive4DtetIds;

    for (size_t i = 0; i < simps.size()/4; ++i) {
        if (tetMarkTags[i] == 1)
        {
            marked_tets.push_back(simps[4*i]);
            marked_tets.push_back(simps[4*i + 1]);
            marked_tets.push_back(simps[4*i + 2]);
            marked_tets.push_back(simps[4*i + 3]);
            markTetActive4DtetIds.push_back(tetActive4DtetIds[i]);
        } else {
            unmarked_tets.push_back(simps[4*i]);
            unmarked_tets.push_back(simps[4*i + 1]);
            unmarked_tets.push_back(simps[4*i + 2]);
            unmarked_tets.push_back(simps[4*i + 3]);

        }
    }

    size_t num_marked   = marked_tets.size();
    size_t num_unmarked = unmarked_tets.size();
    std::cout << " marked tet num " << num_marked << std::endl;
    out.write(reinterpret_cast<const char*>(&num_marked),   sizeof(size_t));
    out.write(reinterpret_cast<const char*>(marked_tets.data()),
              num_marked * sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&num_unmarked), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(unmarked_tets.data()),
              num_unmarked * sizeof(size_t));

    for (size_t i = 0; i < markTetActive4DtetIds.size(); ++i) {
        size_t len = markTetActive4DtetIds[i].size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(markTetActive4DtetIds[i].data()),
                  len * sizeof(size_t));
    }

    out.close();
}
void load_column_mesh_binary(
    std::vector<double>&              verts,
    std::vector<std::vector<double>>& time,
    std::vector<size_t>&              marked_tets,
    std::vector<size_t>&              unmarked_tets,
    const std::string&                filepath)
{
    std::ifstream in(filepath, std::ios::binary);

    size_t num_verts;
    in.read(reinterpret_cast<char*>(&num_verts), sizeof(size_t));

    verts.resize(num_verts * 3);
    in.read(reinterpret_cast<char*>(verts.data()),
            verts.size() * sizeof(double));

    time.resize(num_verts);
    for (size_t i = 0; i < num_verts; ++i) {
        size_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(size_t));
        time[i].resize(len);
        in.read(reinterpret_cast<char*>(time[i].data()),
                len * sizeof(double));
    }

    size_t num_marked;
    in.read(reinterpret_cast<char*>(&num_marked), sizeof(size_t));
    marked_tets.resize(num_marked);
    in.read(reinterpret_cast<char*>(marked_tets.data()),
            num_marked * sizeof(size_t));

    size_t num_unmarked;
    in.read(reinterpret_cast<char*>(&num_unmarked), sizeof(size_t));
    unmarked_tets.resize(num_unmarked);
    in.read(reinterpret_cast<char*>(unmarked_tets.data()),
            num_unmarked * sizeof(size_t));

    in.close();
}