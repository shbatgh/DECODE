mod extract_features;
mod kmeans;

fn main() {
    let features = extract_features::read_outlines("/Users/sam/dev/kmeans/src/9_outlines.txt").expect("Failed to read outlines");
    //println!("{:?}", features[0][0].0);

    let hull = extract_features::convex_hull(features.clone());
    //println!("{:?}", hull[0]);

    for cell in &features {
        println!("{:?} {:?}", extract_features::convex_area(cell), extract_features::convex_perimeter(cell));
    }

    let rgb_matrix = kmeans::load_image_as_matrix("/Users/sam/dev/kmeans/src/9.png");
    let segmentation_rgb = extract_features::load_segmentations_as_matrix("/Users/sam/dev/kmeans/src/9.png", "/Users/sam/dev/kmeans/src/9_outlines.txt");
    //println!("{:?}", extract_features::channel_mean(&segmentation_rgb));
}

// DECODE: DEep Cell Observation & Discovery Engine