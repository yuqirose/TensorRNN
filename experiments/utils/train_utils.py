import tensorflow as tf

def fill_feed_dict(batch, train_phase=True):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        batch: Tuple (Images, labels)
        evaluation: boolean, used to set dropout_rate to 1 in case of evaluation          
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    graph = tf.get_default_graph()
    images_ph = graph.get_tensor_by_name('placeholder/images:0')
    labels_ph = graph.get_tensor_by_name('placeholder/labels:0')
    train_phase_ph = graph.get_tensor_by_name('placeholder/train_phase:0')
    
    images_feed, labels_feed = batch    
    feed_dict = {
        images_ph: images_feed,
        labels_ph: labels_feed,
        train_phase_ph: train_phase
    }    
    return feed_dict