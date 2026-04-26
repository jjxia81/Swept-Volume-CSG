//
//  io.cpp
//  adaptive_colume_grid
//
//  Created by Yiwen Ju on 12/3/24.
//
#include "io.h"
#include <sweep/logger.h>

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
    std::vector<double>& verts,
    std::vector<size_t>& simps,
    std::vector<int>& tetActiveTags,
    std::vector<std::vector<double>>& time,
    std::vector<std::vector<double>>& values,
    bool cyclic)
{
    size_t vert_num = grid.get_num_vertices();
    size_t tet_num = grid.get_num_tets();
    size_t tet4d_num = 0, vert4d_num = 0;
    verts.reserve(vert_num * 3);
    simps.reserve(tet_num * 4);
    tetActiveTags.reserve(tet_num);
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
    size_t active_tet_count = 0;
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
        tet4d_num += vertexMap[value_of(vs[0])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[1])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[2])].vert4dList.size();
        tet4d_num += vertexMap[value_of(vs[3])].vert4dList.size();
        tet4d_num -= 4;
    });

    sweep::logger().info("3D grid tet Number: {} 3D active Tet Number: {}", grid.get_num_tets(), active_tet_count);

    sweep::logger().info("4D Vertex Number: {} 4D Tetrahedra Number: {}", vert4d_num, tet4d_num);
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