package alerter

import (
	"encoding/json"
	"fmt"
)

// ParseConfig parses a JSON config blob into a *Config. JSON is used
// instead of YAML so the alerter binary stays free of a yaml-parser
// dependency; the file extension on disk is .json. The cmd loader
// reads any path ending in .json or unsuffixed.
func ParseConfig(data []byte) (*Config, error) {
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("parse alerter config: %w", err)
	}
	return &c, nil
}
