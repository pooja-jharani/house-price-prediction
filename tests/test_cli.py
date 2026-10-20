from house_price.cli import main


def test_generate_data_command(tmp_path):
    out = tmp_path / "houses.csv"
    assert main(["generate-data", "--n-samples", "120", "--output", str(out)]) == 0
    assert out.exists()


def test_predict_rejects_invalid_input_with_nonzero_exit():
    code = main(["predict", "--area", "5", "--bedrooms", "3", "--bathrooms", "2",
                 "--age", "5", "--distance", "4", "--location", "Suburb"])
    assert code == 1
