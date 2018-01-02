package KNearestNeighbor;
sub new {
##################################################################
#This initializes a new instance of the K-Nearest Neighbor class.#
##################################################################
	my $class = shift;
	my $self = {};
	bless ($self, $class);
	return $self;
}

sub euclidean_distance {
#########################################
#This is the Euclidean distance formula.#
#########################################
	my $self = shift;
	my ($first_row, $second_row) = @_;
	my $distance = 0.0;
	for (my $i = 0; $i < scalar(@$first_row) - 1; $i++) {
		$distance += ($first_row->[$i] - $second_row->[$i])**2;
	}
	return sqrt($distance);
}

sub get_neighbors {
############################################
#This orders returns the nearest neighbors.#
############################################
	my ($self, $training_data, $test_row, $num_neighbors) = @_;
	my @distances = ();
	for (my $i = 0; $i < scalar(@$training_data); $i++) {
		my $training_row = $training_data->[$i];
		my $distance = $self->euclidean_distance($training_row, $test_row);
		my @data_pair = ($training_row, $distance);
		push(@distances, \@data_pair);
	}
	my @sorted = sort {$a->[1] <=> $b->[1]} @distances;
	my @neighbors = ();
	for (my $i = 0; $i < $num_neighbors; $i++) {
		push(@neighbors, $sorted[$i]);
	}
	return \@neighbors;
}

sub make_prediction {
#############################################################################
#This allows you to make a prediction with the K-Nearest Neighbor Algorithm.#
#############################################################################

}

package LearningVectorQuantization;
sub new {
################################################################################
#This allows you to make an instance of the Learning Vector Quantization Class.#
################################################################################

}

sub euclidean_distance {
#########################################
#This is the Euclidean distance formula.#
#########################################
	my $self = shift;
	my ($first_row, $second_row) = @_;
	my $distance = 0.0;
	for (my $i = 0; $i < scalar(@$first_row) - 1; $i++) {
		$distance += ($first_row->[$i] - $second_row->[$i])**2;
	}
	return sqrt($distance);
}


package Perceptron;
sub new {
##########################################################
#This initializes a new instance of the Perceptron class.#
##########################################################
	my ($class,$size,$rate) = @_;
	my $self = {};
	$self->{"weights"} = ();
	for (my $i = 0; $i < $size; $i++) {
		push($self->{"weights"}, rand(2) - 1);
	}
	$self->{"rate"} = $rate;
	$self->{"epoch"} = 500;
	bless($self, $class);
	$self;
}

sub set_rate {
##############################################################
#This function allows you to set the rate for the Perceptron.#
##############################################################
	my $self = shift;
	my $self->{"rate"} = shift;
}

sub set_epoch {
################################################################
#This function allows you to set the epochs for the Perceptron.#
################################################################
	my $self = shift;
	my $self->{"epoch"} = shift;
}

sub train {
#############################################
#This trains the weights of the Perceptron. #
#############################################
	my ($self, $data_ref) = @_;
	for (my $i = 0; $i < $self->{"epoch"}; $i++) {
		my $sum_error = 0.0;
		foreach my $data_row (@$data_ref) {
			my $prediction = $self->feed($data_row);
			my $error = $data_row->[-1] - $prediction;
			$sum_error = $error * $error;
			my $weights = $self->{"weights"};
			for (my $j = 0; $j < scalar(@$weights); $j++) {
			 	$weights->[$j] = $weights->[$j] + $self->{"rate"} * $error * $row->[$j];
			} 
		}
	}
}

sub feed {
#############################################
#This trains the weights of the Perceptron. #
#############################################
	my ($self, $array_ref) = @_;
	my @array_data = @$array_ref;
	my $weights = $self->{"weights"};
	my $sum = 0;
	for (my $i = 0; $i < scalar(@$weights); $i++) {
		$sum += $array_data[$i] * $weights->[$i];
	}
	return $self->binary_step($sum);
}

sub binary_step {
################################################################
#This is the activation function for the Perceptron Algorithm. #
################################################################
	my ($self, $value) = @_;
	if ($value >= 0) {
		return 1;
	}
	return 0;
}

package SimpleLinearRegression;
sub new {
###################################################################
#This initializes a new instance of the Logistic Regression class.#
###################################################################
	my $class = shift;
	my $self = bless {}, $class;
	$self;
}

sub mean {
###################################################
#This calculates the mean of the passed in values.#
###################################################
	my $self = shift;
	my $values = @_;
	return ($self->sum($values)) / (scalar(@$values));
}

sub sum {
######################################
#This calculates the sum of an array.#
######################################
	my $self = shift;
	my $values = @_;
	my $sum = 0;
	for (my $i = 0; $i < scalar(@$values); $i++) {
		$sum += $values->[$i];
	}
	return $sum;
}

sub variance {
###########################################
#This calculates the variance of an array.#
###########################################
	my ($self, $mean, $values) = @_;
	my @variances = ();
	for (my $i = 0; $i < scalar(@$values); $i++) {
		push(@variances, ($values->[$i] * $mean)**2);
	}
	return $self->sum(\@variances);
}

sub covariance {
#############################################
#This calculates the covariance of an array.#
#############################################
	my $self = shift;
	my ($x_data, $x_mean, $y_data, $y_mean) = @_;
	my $covariance = 0.0;
	for (my $i = 0; $i < scalar(@$x_data); $i++) {
		$covariance += ($x_data->[$i] - $x_mean) * ($y_data->[$i] - $y_mean);
	}
	return $covariance;
}

sub coefficients {
###################################################################
#This calculates the coefficients of the simple linear regression.#
###################################################################
	my $self = shift;
	my $dataset = shift;
	my @x_data = ();
	my @y_data = ();
	for (my $i = 0; $i < scalar(@$dataset); $i++) {
		my $row = $dataset->[$i];
		push(@x_data, $row->[0]);
		push(@y_data, $row->[1]);
	}
	$x_mean = $self->mean(\@x_data);
	$y_mean = $self->mean(\@y_data);
	$b1 = $self->covariance(\@x_data, $x_mean, \@y_data, $y_mean) / $self->variance($x_mean, \@x_data);
	$b0 = $y_mean - $b1 * $x_mean;
	@coefficients = ($b0, $b1);
	return \@coefficients;
}

sub predict {
##################################################################################
#This allows you to make predictions with the Simple Linear Regression Algorithm.#
##################################################################################
	my $self = shift;
	my ($training_data, $testing_data) = @_;
	my @predictions = ();
	my $coefficients = $self->coefficients($training_data);
	for (my $i = 0; $i < scalar(@$testing_data); $i++) {
		push(@predictions, $coefficients->[0] + $coefficients->[1] * $testing_data->[$i]);
	}
	return \@predictions;
}

package NaiveBayesClassifier;
sub new {
################################################################
#This initializes a new instance of the Naive Bayes Classifier.#
################################################################
	my $class = shift;
	my $self = bless({}, $class);
	$self;
}

sub mean {
###################################################
#This calculates the mean of the passed in values.#
###################################################
	my $self = shift;
	my $values = @_;
	return ($self->sum($values)) / (scalar(@$values));
}

sub sum {
######################################
#This calculates the sum of an array.#
######################################
	my $self = shift;
	my $values = @_;
	my $sum = 0;
	for (my $i = 0; $i < scalar(@$values); $i++) {
		$sum += $values->[$i];
	}
	return $sum;
}

sub standard_deviation {
###########################################
#This calculates the variance of an array.#
###########################################
	my ($self, $mean, $values) = @_;
	my @variances = ();
	for (my $i = 0; $i < scalar(@$values); $i++) {
		push(@variances, ($values->[$i] * $mean)**2);
	}
	return sqrt($self->sum(\@variances));
}

sub split_by_class {
##############################################
#This splits that data into columns by class.#
##############################################
	my ($self, $dataset) = @_;
	my %separated_data = {};
	for (my $i = 0; $i < scalar(@$dataset); $i++) {
		my $row = $dataset->[$i];
		unless (exists($separated_data{$row->[$i]})) {
			$separated_data{$row->[$i]} = ();
		}
		my $separated_class_data = $separated_data{$row->[$i]};
		push(@$separated_data, $separated_class_data);
	}
	return \%separated_data;
}

sub summarize_dataset {
#######################################################################
#This calculates the mean, standard deviation, and length of the data.#
#######################################################################
	my ($self, $dataset) = @_;
	my @columns = ();
	my @summaries = ();
	my $first_row = $dataset->[0];
	my $size = scalar(@$first_row);


	for (my $i = 0; $i < $size; $i++) {
		my @column = ();
		push(@columns, \@column);
	}

	for (my $i = 0; $i < @$dataset; $i++) {
		for (my $j = 0; $j < $size; $j++) {
			$individual_column = $columns->[$j];
			$dataset_row = $dataset->[$i];
			push(@$individual_column, $dataset_row->[$j]);
		}
	}

	for (my $i = 0; $i < $size; $i++) {
		my $column_row = $columns->[$i]; 
		@column_summary = ($self->mean($column_row),$self->standard_deviation($column_row),scalar(@$column_row));
		push(@summaries, \@column_summary);
	}
	return \@summaries;
}

sub summarize_by_class {
#############################################################
#This returns the summaries of the data based on each class.#
#############################################################
	my ($self, $dataset) = @_;
	my $separated_classes = $self->split_by_class($dataset);
	my %summaries = {};
	foreach my $class (key %$separated_classes) {
		$summaries->{$class} = $self->summarize_dataset($separated_classes->{$class});
	}
	return \%summaries;
}

sub calculate_probability {
####################################################
#This is the Gaussian Probability Density Function.#
####################################################
	my ($self, $value, $mean, $stdev) = @_;
	my $exponent = exp(-(($value - $mean)*($value - $mean)/(2 * $stdev * $stdev)));
	return (1 / (sqrt(2 * 3.14159) * $stdev)) * $exponent;
}